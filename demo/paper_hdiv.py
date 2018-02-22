# Here we solve the problem similar to the one discussed in
# `Mortar finite elements for interface problems`
#
# With \Omega = [-1/4, 5/4]^d and \Omega_2 = [1/4, 3/4]^d the problem reads
#
# -\Delta u_1 + u_1 = f_1 in \Omega \ \Omega_2=\Omega_1
#  \Delta u_2 + u_2 = f_2 in \Omega_2
#  n1.grad(u_1) + n2.grad(u_2) = g on \partial\Omega_2=Gamma
#  u1 - u2 = 0 on \Gamma
#  u1 = 0 in \partial\Omega_1
#
# in the mixed form
from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from fenics_ii.utils.convert import block_diagonal_matrix

from block import block_mat, block_vec, block_bc
from block.algebraic.petsc import LU, InvDiag

from hsmg import HsNormMG
from hsmg.hsquad import BP_H1Norm

from dolfin import *
import numpy as np


def n_generator(mg_levels, nrefs):
    '''n in UnitSquare/UnitCube'''
    # The geom setup here [-1/4, 5/4] makes things a bit awkard
    n0 = 6
    k = 1
    while k < mg_levels:  n0 *= 2

    for _ in range(0, nrefs):
        yield n0
        n0 *= 2

def compute_hierarchy(dim, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0
    
    interior = {2: 'std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25',
                3: 'std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))) < 0.25'}[dim]
    
    interior = CompiledSubDomain(interior)
    if nlevels == 1:
        if dim == 2:
            outer_mesh = RectangleMesh(Point(*(-0.25, )*dim), Point(*(1.25, )*dim), n, n)
        else:
            outer_mesh = BoxMesh(Point(*(-0.25, )*dim), Point(*(1.25, )*dim), n, n, n)

        File('foo.pvd') << outer_mesh
        subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
        for cell in cells(outer_mesh):
            x = cell.midpoint().array()            
            subdomains[cell] = int(interior.inside(x, False))
        assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

        inner_mesh = SubMesh(outer_mesh, subdomains, 1)
        outer_mesh = SubMesh(outer_mesh, subdomains, 0)

        surfaces = MeshFunction('size_t', inner_mesh, dim-1, 0)
        DomainBoundary().mark(surfaces, 1)
        gamma_mesh = EmbeddedMesh(inner_mesh, surfaces, 1, [0.5, ]*dim)
        
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [(outer_mesh, inner_mesh, gamma_mesh)]

    return compute_hierarchy(dim, n, 1) + compute_hierarchy(dim, n/2, nlevels-1)


def setup_system(rhs_data, precond, meshes, mg_params_):
    '''Solver'''
    outer_mesh, inner_mesh, gamma_mesh = meshes[0]

    # Extract botttom edge meshes
    hierarchy = [m[-1] for m in meshes]
        
    # Space of u and the Lagrange multiplier
    S1 = FunctionSpace(outer_mesh, 'RT', 1)
    S2 = FunctionSpace(inner_mesh, 'RT', 1)
    V1 = FunctionSpace(outer_mesh, 'DG', 0)
    V2 = FunctionSpace(inner_mesh, 'DG', 0)
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 0)
    W = [S1, S2, V1, V2, Q]

    sigma1, sigma2, u1, u2, p = map(TrialFunction, W)
    tau1, tau2, v1, v2, q = map(TestFunction, W)

    dxGamma = Measure('dx', domain=gamma_mesh.mesh)
    n_gamma = gamma_mesh.normal('+')          # Outer of inner square
    
    a00 = inner(sigma1, tau1)*dx
    a11 = inner(sigma2, tau2)*dx

    a02 = inner(-u1, div(tau1))*dx
    a13 = inner(-u2, div(tau2))*dx

    a20 = inner(-v1, div(sigma1))*dx
    a31 = inner(-v2, div(sigma2))*dx

    a22 = inner(-u1, v1)*dx
    a33 = inner(-u2, v2)*dx

    # Coupling stuff
    a04 = inner(p, dot(tau1('+'), n_gamma))*dxGamma    
    a14 = inner(p, dot(tau2('+'), n_gamma))*dxGamma    # Sign!
    a40 = inner(q, dot(sigma1('+'), n_gamma))*dxGamma    
    a41 = inner(q, dot(sigma2('+'), n_gamma))*dxGamma    # Sign!

    f1, f2, g = rhs_data
    # Rhs
    dim = outer_mesh.geometry().dim()
    L0 = inner(Constant((0, )*dim), tau1)*dx
    L1 = inner(Constant((0, )*dim), tau2)*dx
    L2 = inner(-f1, v1)*dx
    L3 = inner(-f2, v2)*dx
    L4 = inner(g, q)*dx  # Sign?

    # Assembly
    A00, A11, A02, A13, A20, A31, A22, A33 = map(assemble,
                                                 (a00, a11, a02, a13, a20, a31, a22, a33))

    A04, A14, A40, A41 = [trace_assemble(a, gamma_mesh) for a in (a04, a14, a40, a41)]
    A14 *= -1
    A41 *= -1

    AA = block_mat([[A00, 0,   A02, 0,   A04],
                    [0,   A11, 0,   A13, A14],
                    [A20, 0,   A22, 0,   0  ],
                    [0,   A31, 0,   A33, 0  ],
                    [A40, A41, 0,   0,   0  ]])
    
    bb = block_vec(map(assemble, (L0, L1, L2, L3, L4)))

    # Preconditioner
        # Preconditioner blocks
    P00 = LU(assemble(inner(div(sigma1), div(tau1))*dx + inner(sigma1, tau1)*dx))
    P11 = LU(assemble(inner(div(sigma2), div(tau2))*dx + inner(sigma2, tau2)*dx))
    P22 = InvDiag(assemble(inner(u1, v1)*dx))
    P33 = InvDiag(assemble(inner(u2, v2)*dx))

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)
    
    if precond == 'mg':
        P44 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        P44 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=0.5, as_type=PETScMatrix)            # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P44 = BP_H1Norm(Q, 0.5, bp_params)
    print 'Setup B'

    # The preconditioner
    BB = block_diagonal_matrix([P00, P11, P22, P33, P44])

    return AA, bb, BB, W

    
def setup_case_2d():
    from mms_setups import paper_hdiv_2d
    return paper_hdiv_2d()


def setup_case_3d():
    assert False
    # from mms_setups import paper_hdiv_3d
    # return paper_hdiv_3d()


def setup_error_monitor(true, memory):
    from error_convergence import monitor_error, Hdiv_norm, L2_norm, Hs_norm
    return monitor_error(true, [Hdiv_norm, Hdiv_norm, L2_norm, L2_norm, Hs_norm(0.5)], memory)


def setup_fractionality(): return 0.5
