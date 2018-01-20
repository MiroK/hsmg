# Here we solve the Babuska problem
#   
#   -\Delta u + u = f  in \Omega
#       grad(u).n = g  in \Gamma
#
# A mixed for is used with
#
#     sigma - grad(u) = 0
#     div(sigma) - u = -f
#     sigma.n        = g
#
# So we endup with
#    (sigma, tau) + (u, div(tau))  (p, tau.n) = 0
#    (dig(sigma), v) -(u, v)                   = -(f, v)
#    (sigma.n, q)                             = (g, q)
#
# Enforcing bcs weakly leads to saddle point formulation with Lagrange
# multiplier in H^0.5 requiring Schur complement preconditioner based
# on -\Delta ^ 0.5. Here p = -u on the boundary

from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from block import block_mat, block_vec, block_bc
from block.algebraic.petsc import LU, InvDiag

from hsmg import HsNormMG
from hsmg.hsquad import BP_H1Norm

from dolfin import *
import numpy as np


def setup_system(rhs_data, precond, meshes, mg_params_):
    '''Solver'''
    omega_mesh = meshes[0]
    # Extract botttom edge meshes
    hierarchy = []
    gamma_mesh = None
    for mesh in meshes:
        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
        gdim = mesh.geometry().dim()
        # CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        DomainBoundary().mark(facet_f, 1)
        gmesh = EmbeddedMesh(mesh, facet_f, 1, [0.5, ]*gdim)        

        if gamma_mesh is None: gamma_mesh = gmesh

        hierarchy.append(gmesh.mesh)
        
    if rhs_data is None:
        f, g = Constant(1), Expression('sin(pi*(x[0] + x[1]))', degree=1)
    else:
        f, g = rhs_data

    S = FunctionSpace(omega_mesh, 'RT', 1)        # sigma
    V = FunctionSpace(omega_mesh, 'DG', 0)        # u
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 0)   # p
    W = [S, V, Q]

    sigma, u, p = map(TrialFunction, W)
    tau, v, q = map(TestFunction, W)

    dxGamma = dx(domain=gamma_mesh.mesh)        
    n_gamma = gamma_mesh.normal('+')          # Outer

    # System - for symmetry
    a00 = inner(sigma, tau)*dx
    a01 = inner(u, div(tau))*dx
    a02 = inner(dot(tau('+'), n_gamma), p)*dxGamma
    
    a10 = inner(div(sigma), v)*dx
    a11 = -inner(u, v)*dx # FIXme: double check sign here
    
    a20 = inner(dot(sigma('+'), n_gamma), q)*dxGamma

    L0 = inner(Constant((0, )*gdim), tau)*dx
    L1 = inner(-f, v)*dx
    L2 = inner(g, q)*dxGamma

    A00, A01, A10, A11 = map(assemble, (a00, a01, a10, a11))
    A02 = trace_assemble(a02, gamma_mesh)
    A20 = trace_assemble(a20, gamma_mesh)

    AA = block_mat([[A00, A01, A02],
                    [A10, A11,   0],
                    [A20, 0,     0]])

    bb = block_vec(map(assemble, (L0, L1, L2)))

    print 'Assembled AA'
    # Preconditioner blocks
    P00 = LU(assemble(inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx))
    P11 = InvDiag(assemble(inner(u, v)*dx))

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)
    
    # Trace of Hdiv is H^{-1/2} and the dual is H^{1/2}
    if precond == 'mg':
        P22 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        P22 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=0.5, as_type=PETScMatrix)            # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P22 = BP_H1Norm(Q, 0.5, bp_params)
    print 'Setup B'
        
    # The preconditioner
    BB = block_mat([[P00, 0, 0], [0, P11, 0], [0, 0, P22]])

    return AA, bb, BB, W

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import numpy as np
    from common import log_results, cond_solve, iter_solve, direct_solve
    from babuska_H1 import compute_hierarchy
    
    parser = argparse.ArgumentParser()
    # What
    parser.add_argument('-D', type=int, help='Solve 2d or 3d problem',
                         default=2, choices=[2, 3])
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-Q', type=str, help='iters (with MinRes) or cond (using CGN)',
                        default='iters', choices=['iters', 'cond', 'sane'])
    # How
    parser.add_argument('-B', type=str, help='eig preconditioner or MG preconditioner',
                        default='MG', choices=['eig', 'mg', 'bp'])
    
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                        default='')
    # Iter settings
    parser.add_argument('-tol', type=float, help='Relative tol for Krylov',
                        default=1E-12)
    parser.add_argument('-nlevels', type=int, help='Number of levels for multigrid',
                        default=4)
    parser.add_argument('-eta', type=float, help='eta parameter for MG smoother',
                        default=1.0)
    parser.add_argument('-mes', type=int, help='Macro element size for MG smoother',
                        default=1)
    # Keep an eye on the error of the converged solution
    parser.add_argument('-error', type=int, help='Compare to analytical solution',
                        default=0)

    args = parser.parse_args()

    dim = args.D

    main = {'iters': iter_solve,
            'cond': cond_solve,
            'sane': direct_solve}[args.Q]

    # What rhs to use and monitoring
    if args.error:
        from error_convergence import monitor_error, Hdiv_norm, L2_norm, Hs_norm
        from mms_setups import babuska_Hdiv_2d

        if dim == 2:
            up, fg = babuska_Hdiv_2d()
        else:
            raise NotImplementedError
        
        memory = []
        monitor = monitor_error(up, [Hdiv_norm, L2_norm, Hs_norm(0.5)], memory)
    else:
        memory, fg, monitor = None, None, None

    init_level = 5
    sizes, history = [], []
    for n in [2**i for i in range(init_level, init_level+args.n)]:
        # Embedded
        hierarchy = compute_hierarchy(dim, n, nlevels=args.nlevels)
        print 'Hierarchies', [mesh.num_vertices() for mesh in hierarchy]
        
        setup = setup_system(fg, args.B, hierarchy, mg_params_={'macro_size': args.mes, 'eta': args.eta})
        size, value, u = main(setup, args.tol)

        if monitor is not None: monitor.send(u)

        msg = 'Problem size %d[%s], current %s is %g, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (sum(size), size, args.Q, value, history[::-1]))
        history.append((value, ))
        sizes.append(size)
        
    # S, V, Q and cond or iter
    args.log and log_results(args, sizes, {-0.5: history}, fmt='%d %d %d %.16f', cvrg=memory)
