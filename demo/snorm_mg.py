# Multigrid as preconditioner for eigenvalue laplacians

from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from hsmg import Hs0NormMG, HsNormMG

from block.iterative import ConjGrad

from dolfin import *
import numpy as np


def main(hierarchy, s):
    '''
    Solve Ax = b where A is the eigenvalue representation of (-Delta + I)^s
    '''

    mesh = hierarchy[0]
    V = FunctionSpace(mesh, 'CG', 1)
    
    A = H1_L2_InterpolationNorm(V, s).get_s_norm(s=s, as_type=PETScMatrix)
    
    mg_params = {'macro_size': 1,
                 'nlevels': len(hierarchy),
                 'eta': 0.4}
    # FIXME, bdry = None does not work at the moment
    bdry = DomainBoundary()
    B = HsNormMG(V, bdry, s, mg_params, mesh_hierarchy=hierarchy)  

    x = Function(V).vector()

    xrand = np.random.random(x.local_size())
    xrand -= xrand.sum()/x.local_size()
    x.set_local(xrand)

    b = Function(V).vector()

    Ainv = ConjGrad(A, precond=B, initial_guess=x, tolerance=1e-13, maxiter=500, show=2)

    # Compute solution
    x = Ainv * b

    niters = len(Ainv.residuals) - 1
    size = V.dim()
    
    return size, niters

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('s', type=float, help='Exponent of the operator')
    parser.add_argument('-D', type=int, help='Solve 2d in 1d, or 2d in 3d problem',
                        default=2)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    args = parser.parse_args()

    dim = args.D

    Mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]
    
    gamma = {2: 'near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)',
             3: 'near(std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))), 0.25)'}
    
    gamma = CompiledSubDomain(gamma[dim])
    
    def compute_hierarchy(n, nlevels):
        '''
        The mesh where we want to solve is n. Here we compute previous
        levels for setting up multrid. nlevels in total.
        '''
        assert nlevels > 0

        if nlevels == 1:
            mesh = Mesh(*(n, )*dim)

            markers = FacetFunction('size_t', mesh, 0)
            gamma.mark(markers, 1)
            assert sum(1 for _ in SubsetIterator(markers, 1)) > 0
            # NOTE: !(EmbeddedMesh <:  Mesh)
            return [EmbeddedMesh(mesh, markers, 1, normal=[0.5]*dim).mesh]

        return compute_hierarchy(n, 1) + compute_hierarchy(n/2, nlevels-1)

    history = []
    for n in [2**i for i in range(5, 5+args.n)]:
        hierarchy = compute_hierarchy(n, nlevels=4)

        size, niters = main(hierarchy, s=args.s)

        msg = 'Problem size %d, current iters %d, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (size, niters, history))
        history.append(niters)
