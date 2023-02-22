# Here we solve the problem similar to the one discussed in
# `Mortar finite elements for interface problems`
#
# With \Omega = [-1/4, 5/4]^d and \Omega_2 = [1/4, 3/4]^d the problem reads
#
# -\Delta u_1 + u_1 = f_1 in \Omega \ \Omega_2=\Omega_1
#  \Delta u_2 + u_2 = f_2 in \Omega_2
#  n1.grad(u_1) + n2.grad(u_2) = 0 on \partial\Omega_2=Gamma
#  eps(u1 - u2) + grad(u1).n1 = g on \Gamma
#  grad(u1).n1 = 0 in \partial\Omega_1

from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from block import block_mat, block_vec, block_bc
from block.algebraic.petsc import AMG

from hsmg import HsNormMG, HsNorm, BP_H1Norm

from dolfin import *
import numpy as np


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
            outer_mesh = UnitSquareMesh(n, n)
        else:
            outer_mesh = UnitCubeMesh(n, n, n)

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


def setup_system(rhs_data, precond, meshes, mg_params_, sys_params):
    '''Solver'''
    outer_mesh, inner_mesh, gamma_mesh = meshes[0]

    # Extract botttom edge meshes
    hierarchy = [m[-1].mesh for m in meshes]
        
    # Space of u and the Lagrange multiplier
    V1 = FunctionSpace(outer_mesh, 'CG', 1)
    V2 = FunctionSpace(inner_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'CG', 1)
    W = [V1, V2, Q]

    u1, u2, p = map(TrialFunction, W)
    v1, v2, q = map(TestFunction, W)

    dxGamma = Measure('dx', domain=gamma_mesh.mesh)

    a00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
    a02 = inner(p, v1)*dxGamma

    a11 = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
    a12 = inner(p, v2)*dxGamma

    a20 = inner(q, u1)*dxGamma
    a21 = inner(q, u2)*dxGamma

    eps_ = sys_params['eps']
    a22 = Constant(-1./eps_)*inner(p, q)*dxGamma
    f1, f2, g = rhs_data

    # NOTE: there is specific assumption of zero Neumann bcs
    L0 = inner(f1, v1)*dx
    # And also interface flux continuity
    L1 = inner(f2, v2)*dx
    L2 = inner(g*Constant(1./eps_), q)*dxGamma

    # Blocks
    A00, A11, A22 = map(assemble, (a00, a11, a22))
    A02, A12 = [trace_assemble(a, gamma_mesh) for a in (a02, a12)]
    A12 *= -1
    A20, A21 = [trace_assemble(a, gamma_mesh) for a in (a20, a21)]
    A21 *= -1

    bb = map(assemble, (L0, L1, L2))
    
    # System
    AA = block_mat([[A00, 0, A02],
                    [0, A11, A12],
                    [A20, A21, A22]])
    bb = block_vec(bb)

    print 'Assembled AA'
    # Preconditioner blocks
    P00 = AMG(A00)
    P11 = AMG(A11)

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)
    
    # Trace of H^1 is H^{1/2} and the dual is H^{-1/2}
    if precond == 'mg':
        P22 = HsNormMG(Q, bdry, -0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        P22 = HsNorm(Q, s=-0.5)**-1
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P22 = BP_H1Norm(Q, -0.5, bp_params)
    print 'Setup B'
        
    # The preconditioner
    BB = block_mat([[P00, 0, 0], [0, P11, 0], [0, 0, P22]])

    return AA, bb, BB, W


def setup_case_2d(**kwargs):
    from mms_setups import paper_mortar_2d
    return paper_mortar_2d(eps=kwargs.get('eps'))


def setup_case_3d(**kwargs):
    from mms_setups import paper_mortar_3d
    return paper_mortar_3d(eps=kwargs.get('eps'))


def setup_error_monitor(true, memory):
    from error_convergence import monitor_error, H1_norm, Hs_norm
    # Note we produce u1, u2, and p error. It is more natural to have
    # broken H1 norm so reduce the first 2 errors to single number
    reduction = lambda e: None if e is None else [sqrt(e[0]**2 + e[1]**2), e[-1]]
    
    return monitor_error(true, [H1_norm, H1_norm, Hs_norm(-0.5)], memory, reduction)


def setup_fractionality(): return -0.5
