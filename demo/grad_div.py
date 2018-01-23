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
        f_rhs, g_rhs = Constant((1, )*gdim), Expression('sin(pi*(x[0] + x[1]))', degree=1)
    else:
        f_rhs, g_rhs = rhs_data

    S = FunctionSpace(omega_mesh, 'RT', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 0)
    W = [S, Q]
    
    sigma, p = map(TrialFunction, W)
    tau, q = map(TestFunction, W)

    dxGamma = dx(domain=gamma_mesh.mesh)        
    n_gamma = gamma_mesh.normal('+')          # Outer
    
    a00 = inner(div(sigma), div(tau))*dx + inner(sigma, tau)*dx
    a01 = inner(dot(tau('+'), n_gamma), p)*dxGamma
    a10 = inner(dot(sigma('+'), n_gamma), q)*dxGamma

    L0 = inner(f_rhs, tau)*dx
    L1 = inner(g_rhs, q)*dxGamma

    A00 = assemble(a00)
    A01 = trace_assemble(a01, gamma_mesh)
    A10 = trace_assemble(a10, gamma_mesh)

    AA = block_mat([[A00, A01],
                    [A10,   0]])

    bb = block_vec(map(assemble, (L0, L1)))

    P00 = LU(A00)

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params)

    if precond == 'mg':
        P11 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        # Bonito
        P11 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=0.5, as_type=PETScMatrix)
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P11 = BP_H1Norm(Q, 0.5, bp_params)
        
    # The preconditioner
    BB = block_mat([[P00, 0], [0, P11]])

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
        from mms_setups import grad_div_2d, grad_div_3d

        if dim == 2:
            up, fg = grad_div_2d()
        else:
            up, fg = grad_div_3d()
        
        memory = []
        monitor = monitor_error(up, [Hdiv_norm, Hs_norm(0.5)], memory)
    else:
        memory, fg, monitor = None, None, None

    init_level = 2 if args.Q == 'sane' else args.nlevels
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
    args.log and log_results(args, sizes, {0.5: history}, fmt='%d %d %.16f', cvrg=memory)
