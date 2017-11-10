# Multigrid as preconditioner for eigenvalue laplacians

from fenics_ii.utils.norms import H10_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from hsmg import Hs0NormMG

from block.iterative import ConjGrad

from dolfin import *
import numpy as np


def main(hierarchy, s):
    '''
    Solve Ax = b where A is the eigenvalue representation of (-Delta + I)^s
    '''
    mesh = hierarchy[0]
    V = FunctionSpace(mesh, 'CG', 1)
    bdry = DomainBoundary()
    
    bcs = DirichletBC(V, Constant(0), bdry)
    A = H10_L2_InterpolationNorm(V, bcs=bcs).get_s_norm(s=s, as_type=PETScMatrix)
    
    mg_params = {'macro_size': 1,
                 'nlevels': len(hierarchy),
                 'eta': 1.0}
    
    # FIXME, bdry = None does not work at the moment
    B = Hs0NormMG(V, bdry, s, mg_params, mesh_hierarchy=hierarchy)  

    x = Function(V).vector()
    # Init guess is random
    xrand = np.random.random(x.local_size())
    xrand -= xrand.sum()/x.local_size()
    x.set_local(xrand)
    bcs.apply(x)
    
    # Zero
    b = Function(V).vector()
    Ainv = ConjGrad(A, precond=B, initial_guess=x, tolerance=1e-13, maxiter=500, show=2)

    # Compute solution
    x = Ainv * b

    # plot(Function(V, x), interactive=True)
    
    niters = len(Ainv.residuals) - 1
    size = V.dim()

    lmin, lmax = np.sort(np.abs(Ainv.eigenvalue_estimates()))[[0, -1]]
    cond = lmax/lmin
    
    return size, niters, cond

# --------------------------------------------------------------------

if __name__ == '__main__':
    from common import log_results, compute_hierarchy
    from os.path import basename
    import argparse
    import re

    D_options = ['1', '2',  # Xd in Xd
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

    # Domain series
    domains = D_options if args.D == 'all' else [args.D]

    for D in domains:
        print '\t\033[1;37;32m%s\033[0m' % D
        for s in s_values:
            print '\t\t\033[1;37;34m%s\033[0m' % s
            history, sizes = [], []
            for n in [2**i for i in range(5, 5+args.n)]:
                hierarchy = compute_hierarchy(D, n, nlevels=4)

                size, niters, cond = main(hierarchy, s=s)

                msg = 'Problem size %d, current iters %d, cond %g, previous %r'
                print '\033[1;37;31m%s\033[0m' % (msg % (size, niters, cond, history[::-1]))
                history.append((niters, cond))
                sizes.append((size, ))
            # Change make header aware properly
            args.s = s
            args.D = D
            args.log and log_results(args, sizes, history, name=basename(__file__),
                                     fmt='%d %d %g')
