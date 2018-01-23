# This is a driver for babuska_H1, babuska_Hdiv, grad_div and curl_curl
# demos
from dolfin import UnitSquareMesh, UnitCubeMesh
from runpy import run_module


def compute_hierarchy(dim, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0
    mesh_init = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]
    
    if nlevels == 1:
        mesh = mesh_init(*(n, )*dim)
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [mesh]

    return compute_hierarchy(dim, n, 1) + compute_hierarchy(dim, n/2, nlevels-1)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import log_results, cond_solve, iter_solve, direct_solve
    import argparse, os
    import numpy as np
    
    parser = argparse.ArgumentParser()
    # The demo file: runnning it defines setups*
    parser.add_argument('demo', type=str, help='Which demo to run')
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
    # Fine tune Krylov
    parser.add_argument('-randomic', type=int, help='Use random initial condition for Krylov or zero?',
                        choices=[0, 1], default=1)
    parser.add_argument('-relconv', type=int, help='Is the tolerance relative',
                        choices=[0, 1], default=1)
    parser.add_argument('-minres', type=str, help='Which MinRes implementation to use',
                        choices=['herzog', 'block', 'petsc'], default='block')
    # Keep an eye on the error of the converged solution
    parser.add_argument('-error', type=int, help='Compare to analytical solution',
                        default=1)

    args = parser.parse_args()

    # The setups
    module, _ = os.path.splitext(args.demo)
    module = __import__(module)  # not importlib in python2.7

    # Config
    dim = args.D

    main = {'iters': iter_solve,
            'cond': cond_solve,
            'sane': direct_solve}[args.Q]

    # What rhs to use and monitoring
    if args.error:
        if dim == 2:
            up, fg = module.setup_case_2d()
        else:
            up, fg = module.setup_case_3d()
        
        memory = []
        monitor = module.setup_error_monitor(up, memory)
    else:
        memory, fg, monitor = None, None, None

    # Multigrid needs a buffer to be able to create coarses meshes
    if args.B != 'mg':
        args.nlevels = 1
    init_level = min(2 , args.nlevels)
    
    sizes, history = [], []
    for level, n in enumerate([2**i for i in range(init_level, init_level+args.n)], 1):
        # Embedded
        hierarchy = compute_hierarchy(dim, n, nlevels=args.nlevels)
        print 'Hierarchies', [mesh.num_vertices() for mesh in hierarchy]
        
        setup = module.setup_system(fg, args.B, hierarchy, mg_params_={'macro_size': args.mes, 'eta': args.eta})
        size, value, u = main(setup, {'tolerance': args.tol,
                                      'randomic': bool(args.randomic),
                                      'relativeconv': bool(args.relconv),
                                      'which_minres': args.minres})

        if monitor is not None: monitor.send(u)

        msg = '(%d/%d) Problem size %d[%s], current %s is %g, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (level, args.n, sum(size), size, args.Q, value, history[::-1]))
        history.append((value, ))
        sizes.append(size)
        
    hs_fract = module.setup_fractionality()
    # dim of spaces + quantity of interest
    args.log and log_results(args, sizes, {hs_fract: history},
                             fmt=' '.join(['%d']*len(up) + ['%.16f']),
                             cvrg=memory)
