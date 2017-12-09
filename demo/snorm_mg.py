# Multigrid as preconditioner for eigenvalue laplacians

from fenics_ii.utils.norms import H1_L2_InterpolationNorm

from hsmg import HsNormMG

from block.iterative import ConjGrad
from block.algebraic.petsc import LumpedInvDiag
from block.algebraic.petsc import LU

from dolfin import *
import numpy as np

import matplotlib.pyplot as plt
fig, ax = plt.subplots()


def generator(hierarchy, tolerance, mg_params_):
    '''
    Solve Ax = b where A is the eigenvalue representation of (-Delta + I)^s
    '''
    
    mesh = hierarchy[0]
    V = FunctionSpace(mesh, 'CG', 1)
    
    As = H1_L2_InterpolationNorm(V)

    bdry = None    
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)
                 

    make_B = lambda s: HsNormMG(V, bdry, s, mg_params, mesh_hierarchy=hierarchy)

    f = Expression('sin(k*pi*x[0])', k=1, degree=4)
    # Wait for s to be send in
    while True:
        s = yield

        A = As.get_s_norm(s=s, as_type=PETScMatrix)
        B = make_B(s)

        x = Function(V).vector()
        # Init guess is random
        xrand = np.random.random(x.local_size())
        xrand -= xrand.sum()/x.local_size()
        x.set_local(xrand)
    
        # Rhs
        v = TestFunction(V)
        b = assemble(inner(f, v)*dx)
    
        Ainv = ConjGrad(A, precond=B, initial_guess=x,
                        tolerance=tolerance, maxiter=500, show=2, relativeconv=True)

        # Compute solution
        x = Ainv * b

        niters = len(Ainv.residuals) - 1
        size = V.dim()

        eigws = np.abs(Ainv.eigenvalue_estimates())
        ax.plot(eigws, label='%d' % V.dim(), linestyle='none', marker='x')
        
        lmin, lmax = np.sort(eigws)[[0, -1]]
        cond = lmax/lmin
    
        yield (size, niters, cond)

        
def compute(gen, s):
    '''Trigger computations for given s'''
    gen.next()
    return gen.send(s)
   
# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import log_results, compute_hierarchy
    from collections import defaultdict
    from os.path import basename
    import argparse
    import re

    # python snorm_mg.py "-0.6 : 0.01 : -0.4" -D 1 -n 7 -log ./results/h1.txt
    
    D_options = ['1', '2',  # Xd in Xd
                 '12', '13', '23',  # Xd in Yd no intersect
                 '-12', '-13', '-23',  # Xd in Yd isect
                 '012', '013', '023']  # Xd in Yd loop
    
    parser = argparse.ArgumentParser()
    # What
    parser.add_argument('s', help='Exponent of the operator or start:step:end')
    parser.add_argument('-D', type=str, help='Solve Xd in Yd problem',
                        default='1', choices=['all'] + D_options)  # Xd in Yd loop
    # How many
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    # Storing
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                         default='')
    # Iterative solver setup
    parser.add_argument('-tol', type=float, help='Relative tol for Krylov',
                         default=1E-12)
    parser.add_argument('-eta', type=float, help='eta parameter for MG smoother',
                         default=1.0)
    parser.add_argument('-mes', type=int, help='Macro element size for MG smoother',
                        default=1)

    args = parser.parse_args()

    # Fractionality series
    print args.s
    try:
        s_values = [float(args.s)]
    except ValueError:
        # Parse linspace
        start, step, end = map(float, re.compile(r'[+-]?\d+(?:\.\d+)?').findall(args.s))
        # FIXME: map needed bacause of silly types in fenicsii
        s_values = map(float, np.arange(start, end+step, step))

    # Domain series
    domains = D_options if args.D == 'all' else [args.D]

    for D in domains:
        print '\t\033[1;37;32m%s\033[0m' % D
        results = defaultdict(list)
        sizes = []
        for level, n in enumerate([2**i for i in range(5, 5+args.n)], 1):
            print '\t\t\033[1;37;31m%s\033[0m' % ('level %d, size %d' % (level, n+1))
            hierarchy = compute_hierarchy(D, n, nlevels=4)
            gen = generator(hierarchy, tolerance=args.tol, mg_params_={'macro_size': args.mes,
                                                                       'eta': args.eta})

            for s in s_values:
                size, niters, cond = compute(gen, s=s)

    
                msg = '@s = %g, Current iters %d, cond %.2f, previous %r'
                print '\033[1;37;34m%s\033[0m' % (msg % (s, niters, cond, results[s][::-1]))
                results[s].append((niters, cond))
            sizes.append((size, ))

            # Log after each n in case of kill signal for large n
            # Change make header aware properly
            if args.log:
                args.D = D
                log_results(args, sizes, results, name=basename(__file__),
                            fmt='%d %d %.16f')

        plt.legend(loc='best')
        plt.show()

