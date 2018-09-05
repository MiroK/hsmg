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


from block.algebraic.petsc import AMG
from block import block_transpose
from hsmg import HsNormMG, HsNorm, BP_H1Norm
# Using new fenics_ii https://github.com/MiroK/fenics_ii
from xii import *  
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
        gamma_mesh = EmbeddedMesh(surfaces, 1)
        
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [(outer_mesh, inner_mesh, gamma_mesh)]

    return compute_hierarchy(dim, n, 1) + compute_hierarchy(dim, n/2, nlevels-1)


def setup_system(rhs_data, precond, meshes, mg_params_, sys_params):
    '''Solver'''
    outer_mesh, inner_mesh, gamma_mesh = meshes[0]

    # Extract botttom edge meshes
    hierarchy = [m[-1] for m in meshes]
        
    # Space of u and the Lagrange multiplier
    V1 = FunctionSpace(outer_mesh, 'CG', 1)
    V2 = FunctionSpace(inner_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh, 'CG', 1)
    W = [V1, V2, Q]

    u1, u2, p = map(TrialFunction, W)
    v1, v2, q = map(TestFunction, W)
    # We will need traces of the functions on the boundary
    Tu1, Tu2 = map(lambda x: Trace(x, gamma_mesh), (u1, u2))
    Tv1, Tv2 = map(lambda x: Trace(x, gamma_mesh), (v1, v2))

    dxGamma = Measure('dx', domain=gamma_mesh)

    a00 = inner(grad(u1), grad(v1))*dx + inner(u1, v1)*dx
    a01 = 0
    a02 = inner(p, Tv1)*dxGamma

    a10 = 0
    a11 = inner(grad(u2), grad(v2))*dx + inner(u2, v2)*dx
    a12 = -inner(p, Tv2)*dxGamma

    a20 = inner(q, Tu1)*dxGamma
    a21 = -inner(q, Tu2)*dxGamma

    eps_ = sys_params['eps']
    assert eps_ > 1
    a22 = Constant(-1./eps_)*inner(p, q)*dxGamma
    f1, f2, g = rhs_data

    # NOTE: there is specific assumption of zero Neumann bcs
    L0 = inner(f1, v1)*dx
    # And also interface flux continuity
    L1 = inner(f2, v2)*dx
    L2 = inner(g*Constant(1./eps_), q)*dxGamma

    a = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    L = [L0, L1, L2]

    AA, bb = map(ii_assemble, (a, L))
    # Make sure that trace blocks map from primal to primal
    # AA[2, 0] is M*T - remove M
    assert len(AA[2][0].chain) == 2
    AA[2][0] = AA[2][0].chain[-1]  # The trace matrix
    assert len(AA[2][1]) == 2
    AA[2][1] = -1*AA[2][1].chain[-1]  # The trace matrix, the sign was with M
    # Block(T)*M
    AA[0][2] = block_transpose(AA[2][0])
    AA[1][2] = block_transpose(AA[2][1])

    # If last row is primal to primal then rhs of Q must be in primal
    # Don't forget to scale eps the interpolant
    bb2 = interpolate(g, Q).vector()
    bb2 *= 1./eps_
    bb[2] = bb2
    
    # Preconditioner blocks
    P00 = AMG(AA[0][0])
    P11 = AMG(AA[1][1])

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)

    # FIXME: special MinRes
    # FIXME: adjust mapping props here
    if precond == 'mg':
        P22 = HsNormMG(Q, bdry, -0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        P22 = HsNorm(Q, s=-0.5)**-1
    else:
        assert False
    print 'Setup B'
        
    # The preconditioner
    BB = block_diag_mat([P00, P11, P22])

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
