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
                 'eta': 0.4}
    
    # FIXME, bdry = None does not work at the moment
    B = Hs0NormMG(V, bdry, s, mg_params, mesh_hierarchy=hierarchy)  

    x = Function(V).vector()
    # Init guess is random
    xrand = np.random.random(x.local_size())
    xrand -= xrand.sum()/x.local_size()
    x.set_local(xrand)
    
    # Zero
    b = Function(V).vector()
    Ainv = ConjGrad(A, precond=B, initial_guess=x, tolerance=1e-13, maxiter=500, show=2)

    # Compute solution
    x = Ainv * b

    niters = len(Ainv.residuals) - 1
    size = V.dim()

    lmin, lmax = np.sort(Ainv.eigenvalue_estimates())[[0, -1]]
    cond = lmax/lmin
    
    return size, niters, cond

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('s', type=float, help='Exponent of the operator')
    parser.add_argument('-D', type=int, help='Solve 1d-1d, 2d-2d, 3d-3d',
                        default=1)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    args = parser.parse_args()

    dim = args.D

    # Embedding mesh
    Mesh = {1: UnitIntervalMesh, 2: UnitSquareMesh, 3: UnitCubeMesh}[dim]

    def compute_hierarchy(n, nlevels):
        '''
        The mesh where we want to solve is n. Here we compute previous
        levels for setting up multrid. nlevels in total.
        '''
        assert nlevels > 0

        if nlevels == 1: return [Mesh(*(n, )*dim)]

        return compute_hierarchy(n, 1) + compute_hierarchy(n/2, nlevels-1)

    history = []
    for n in [2**i for i in range(5, 5+args.n)]:
        hierarchy = compute_hierarchy(n, nlevels=4)
        
        size, niters, cond = main(hierarchy, s=args.s)

        msg = 'Problem size %d, current iters %d, cond %g, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (size, niters, cond, history))
        history.append((niters, cond))

