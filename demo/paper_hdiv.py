from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from fenics_ii.utils.convert import block_diagonal_matrix

from block import block_mat, block_vec, block_bc, block_assemble
from block.algebraic.petsc import LU, InvDiag

from hsmg.hsmg import HsNormMG
from hsmg.hseig import HsNorm

from dolfin import *
import numpy as np

        
def compute_hierarchy(dim, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0

    # Interface between interior and exteriorn domains
    gamma = {2: 'near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)',
             3: 'near(std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))), 0.25)'}
    
    # Marking interior domains
    interior = {2: 'std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25 ? 1: 0',
                3: 'std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))) < 0.25 ? 1: 0'}

    mesh_init = {2: UnitSquareMesh, 3: UnitCubeMesh}
    
    if nlevels == 1:
        if n != 2:
            mesh = mesh_init[dim](*(n, )*dim)
        else:
            if dim == 2:
                mesh = RectangleMesh(Point(0.25, 0.25), Point(0.75, 0.75), 1, 1)
            else:
                mesh = BoxMesh(Point(0.25, 0.25, 0.25), Point(0.75, 0.75, 0.75), 1, 1, 1)

        subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        interior_ = CompiledSubDomain(interior[dim])
        for cell in cells(mesh):
            subdomains[cell] = interior_.inside(cell.midpoint().array(), False)
        mesh.subdomains = subdomains

        markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        CompiledSubDomain(gamma[dim]).mark(markers, 1)
        #DomainBoundary().mark(markers, 1)
        assert sum(1 for _ in SubsetIterator(markers, 1)) > 0
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [(mesh, EmbeddedMesh(mesh, markers, 1, normal=[0.5]*dim))]

    return compute_hierarchy(dim, n, 1) + compute_hierarchy(dim, n/2, nlevels-1)


def setup_system(rhs_data, precond, meshes, mg_params_, sys_params):
    '''Solver'''
    omega, gamma = meshes[0]

    # Extract botttom edge meshes
    hierarchy = [m[-1].mesh for m in meshes]
    f1, f2, g = rhs_data
    # F1 is like u1

    S = FunctionSpace(omega, 'RT', 1)        # sigma
    V = FunctionSpace(omega, 'DG', 0)        # u
    Q = FunctionSpace(gamma.mesh, 'DG', 0)   # p
    W = [S, V, Q]

    sigma, u, p = map(TrialFunction, W)
    tau, v, q = map(TestFunction, W)

    dX = Measure('dx', domain=omega, subdomain_data=omega.subdomains)
    dxGamma = dx(domain=gamma.mesh)
        
    n_gamma = gamma.normal
    n_gamma.vector()[:] *= -1  # Make it consistent with the mms case.
    n_gamma = n_gamma('+')          
    
    # System - for symmetry
    a00 = inner(Constant(1.)*sigma, tau)*dX(0) + inner(Constant(1.)*sigma, tau)*dX(1)
    a01 = -inner(u, div(tau))*dX
    a02 = inner(dot(tau('+'), n_gamma), p)*dxGamma

    a10 = -inner(div(sigma), v)*dX
    # FIXme: double check sign here
    a11 = -inner(u, v)*dX

    a20 = inner(dot(sigma('+'), n_gamma), q)*dxGamma
    a22 = -Constant(sys_params['eps'])*inner(p, q)*dxGamma   

    A00, A01, A10, A11, A22 = map(assemble, (a00, a01, a10, a11, a22))
    A02 = trace_assemble(a02, gamma)
    A20 = trace_assemble(a20, gamma)

    AA = block_mat([[A00, A01, A02],
                    [A10, A11,   0],
                    [A20, 0,   A22]])

    dim = omega.geometry().dim()
    bb = block_assemble([inner(Constant((0, )*dim), tau)*dx,
                         inner(-f1, v)*dX(0) + inner(-f2, v)*dX(1),
                         inner(-g, q)*dxGamma])

    # Block of Riesz preconditioner
    B00 = LU(assemble(inner(sigma, tau)*dX + inner(div(sigma), div(tau))*dX))
    B11 = InvDiag(assemble(inner(u, v)*dX))

    # (Miro) Gamma here is closed loop so H1_L2_Interpolation norm
    # uses eigenalue problem (-Delta + I) u = lambda I u. Also, no
    # boundary conditions are set
    if precond == 'eig':
        B22 = HsNorm(Q, s=0.5)**-1
    elif precond ==  'mg':
        # Alternative B22 block:
        mg_params = {'nlevels': len(hierarchy)}
        mg_params.update(mg_params_)

        bdry = None
        B22 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
    
        B22 = BP_H1Norm(Q, 0.5, bp_params)

    BB = block_mat([[B00, 0, 0],
                    [0, B11, 0],
                    [0, 0, B22]])

    return AA, bb, BB, W


def transform(hierarchy, u):
    '''Break sigma, u, p into pieces on subdomains'''
    mesh, _ = hierarchy[0]

    outer_mesh = SubMesh(mesh, mesh.subdomains, 0)  # 1
    inner_mesh = SubMesh(mesh, mesh.subdomains, 1)  # 2

    sigma_h, u_h, p_h = u

    sigma_elm = sigma_h.function_space().ufl_element()
    S1 = FunctionSpace(outer_mesh, sigma_elm)
    S2 = FunctionSpace(inner_mesh, sigma_elm)
    sigma1_h = interpolate(sigma_h, S1)
    sigma2_h = interpolate(sigma_h, S2)

    u_elm = u_h.function_space().ufl_element()
    V1 = FunctionSpace(outer_mesh, u_elm)
    V2 = FunctionSpace(inner_mesh, u_elm)
    u1_h = interpolate(u_h, V1)
    u2_h = interpolate(u_h, V2)

    # p_h.vector()[:] *= -1.

    return [sigma1_h, sigma2_h, u1_h, u2_h, p_h]


def setup_case_2d(**kwargs):
    from mms_setups import paper_hdiv_2d
    return paper_hdiv_2d(eps=kwargs.get('eps'))


def setup_case_3d(**kwargs):
    from mms_setups import paper_hdiv_3d
    return paper_hdiv_3d(eps=kwargs.get('eps'))


def setup_error_monitor(true, memory):
    from error_convergence import monitor_error, Hdiv_norm, L2_norm, Hs_norm
    # Note we produce sigma1, sigma2 u1, u2, and p error. But this is just
    # so that it is easier to compute error as the true solution is
    # discontinuous. So we compute on subdomain and then reduce
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2),  # Hdiv
                                                  sqrt(e[2]**2 + e[3]**2),  # L2
                                                  e[-1]]

    return monitor_error(true,
                         [Hdiv_norm, Hdiv_norm, L2_norm, L2_norm, Hs_norm(0.5)],
                         memory,
                         reduction)

def setup_fractionality(): return 0.5
