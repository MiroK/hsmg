# Multigrid as preconditioner for eigenvalue laplacians

from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from hsmg.hsquad import BP_H1Norm
from hsmg.utils import to_csr_matrix
from block.iterative import ConjGrad

from petsc4py import PETSc
from dolfin import *
import numpy as np


def generator(mesh, get_k):
    '''
    Solve Ax = b where A is the eigenvalue representation of (-Delta)^s
    '''
    V = FunctionSpace(mesh, 'CG', 1)
    
    As = H1_L2_InterpolationNorm(V)
  
    bp_params = {'k': get_k,
                 'solver': 'cholesky'}
    
    f = Expression('sin(k*pi*x[0])', k=1, degree=4)
    # Wait for s to be send in
    while True:
        s = yield

        A = As.get_s_norm(s=s, as_type=PETScMatrix)
        B = BP_H1Norm(V, s, bp_params)

        x = Function(V).vector()
        # Init guess is random
        xrand = np.random.random(x.local_size())
        xrand -= xrand.sum()/x.local_size()
        x.set_local(xrand)
    
        # Rhs
        v = TestFunction(V)
        b = assemble(inner(f, v)*dx)
    
        Ainv = ConjGrad(A, precond=B, initial_guess=x, tolerance=1e-13, maxiter=500, show=2)

        # Compute solution
        x = Ainv * b

        niters = len(Ainv.residuals) - 1
        size = V.dim()

        eigws = np.abs(Ainv.eigenvalue_estimates())
        lmin, lmax = np.sort(eigws)[[0, -1]]
        cond = lmax/lmin

        nsolves, niters_per_solve = B.nsolves, float(B.niters)/B.nsolves
        k = get_k(s, V.dim(), mesh.hmin())
        print 'with k = %g, solves and niterations per solve[%d(%.4f)]' % (k, nsolves, niters_per_solve)
    
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

    D_options = ['1', '2',  # Xd in Xd
                 '012', '013', '023',
                 '12', '13', '23',  # Xd in Yd no intersect
                 '-12', '-13', '-23']  # Xd in Yd isect
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('s', help='Exponent of the operator or start:step:end')
    parser.add_argument('-D', type=str, help='Solve Xd in Yd problem',
                        default='1', choices=['all'] + D_options)  # Xd in Yd loop
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                         default='')
    args = parser.parse_args()

    # Fractionality series
    try:
        s_values = [float(args.s)]
    except ValueError:
        # Parse linspace
        start, step, end = map(float, re.compile(r'[+-]?\d+(?:\.\d+)?').findall(args.s))
        # FIXME: map needed bacause of silly types in fenicsii
        s_values = map(float, np.arange(start, end+step, step))

    get_k = lambda s, N, h: 5.0*1./ln(N)
    # Domain series
    domains = D_options if args.D == 'all' else [args.D]

    for D in domains:
        print '\t\033[1;37;32m%s\033[0m' % D
        results = defaultdict(list)
        sizes = []
        for level, n in enumerate([2**i for i in range(5, 5+args.n)], 1):
            print '\t\t\033[1;37;31m%s\033[0m' % ('level %d, size %d' % (level, n+1))
            mesh = compute_hierarchy(D, n, nlevels=1)[0]
            gen = generator(mesh, get_k)

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
                log_results(args, sizes, results, name=basename(__file__), fmt='%d %d %g')

    # FiXME: record also the iterations
