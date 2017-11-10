# Multigrid as preconditioner for eigenvalue laplacians

from fenics_ii.utils.norms import H1_L2_InterpolationNorm

from hsmg import HsNormMG

from block.iterative import ConjGrad

from dolfin import *
import numpy as np


def main(hierarchy, s):
    '''
    Solve Ax = b where A is the eigenvalue representation of (-Delta + I)^s
    '''
    mesh = hierarchy[0]
    V = FunctionSpace(mesh, 'CG', 1)
    
    A = H1_L2_InterpolationNorm(V).get_s_norm(s=s, as_type=PETScMatrix)
    
    mg_params = {'macro_size': 1,
                 'nlevels': len(hierarchy),
                 'eta': 1.0}
    # FIXME, bdry = None does not work at the moment
    bdry = None
    B = HsNormMG(V, bdry, s, mg_params, mesh_hierarchy=hierarchy)  

    x = Function(V).vector()
    # Init guess is random
    xrand = np.random.random(x.local_size())
    xrand -= xrand.sum()/x.local_size()
    x.set_local(xrand)

    # Rhs
    v = TestFunction(V)
    f = Expression('sin(k*pi*x[0])', k=1, degree=1)
    b = assemble(inner(f, v)*dx)
    # b = Function(V).vector()

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

    parser = argparse.ArgumentParser()
    
    parser.add_argument('s', help='Exponent of the operator or start:step:end')
    parser.add_argument('-D', type=str, help='Solve Xd in Yd problem',
                        default='1', choices=['1', '2', '3',         # Xd in Xd
                                              '12', '13', '23',      # Xd in Yd no isect
                                              '-12', '-13', '-23',   # Xd in Yd isect
                                              '012', '013', '023'])  # Xd in Yd loop
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                         default='')
    args = parser.parse_args()

    try:
        s_values = [float(args.s)]
    except ValueError:
        # Parse linspace
        start, step, end = map(float, re.compile(r'[+-]?\d+(?:\.\d+)?').findall(args.s))
        # FIXME: map needed bacause of silly types in fenicsii
        s_values = map(float, np.arange(start, end+step, step))
        
    for s in s_values:
        print '\t\033[1;37;34m%s\033[0m' % s
        history, sizes = [], []
        for n in [2**i for i in range(5, 5+args.n)]:
            hierarchy = compute_hierarchy(args.D, n, nlevels=4)

            size, niters, cond = main(hierarchy, s=s)

            msg = 'Problem size %d, current iters %d, cond %g, previous %r'
            print '\033[1;37;31m%s\033[0m' % (msg % (size, niters, cond, history[::-1]))
            history.append((niters, cond))
            sizes.append((size, ))
        # Change s make header aware of that
        args.s = s
        args.log and log_results(args, sizes, history, name=basename(__file__),
                                 fmt='%d %d %g')
