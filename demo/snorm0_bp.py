# Multigrid as preconditioner for eigenvalue laplacians

from fenics_ii.utils.norms import H10_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from hsmg.hsquad import BP_H10Norm, BP_Operator_Base
from hsmg.utils import to_csr_matrix
from block.iterative import ConjGrad

from petsc4py import PETSc
from dolfin import *
import numpy as np

import matplotlib.pyplot as plt
fig, ax = plt.subplots()


class H10_FAST(object):
    '''
    Compute H10 s norm using eigenpairs of continuous Lagrange 1 approx of 
        
        (u', v') = lambda * (u, v) on H10((0, 1))
    '''
    def __init__(self, V, bcs):
        assert V.ufl_element().family() == 'Lagrange'
        assert V.ufl_element().degree() == 1
        assert V.ufl_cell() == interval

        time = Timer('gevp')
        time.start()
        
        bc = DirichletBC(V, Constant(0), 'on_boundary')

        n = V.dim()-1
        k = np.arange(1, n)
        h = V.mesh().hmin()
    
        y = np.cos(k*pi/n)
        lmbda0 = (6./h**2)*(1. - y)/(2. + y)

        xj = V.tabulate_dof_coordinates()

        # Bc eigenvectors
        U = np.zeros((n+1, n+1))
        for i, dof in enumerate(bc.get_boundary_values().keys()):
            U[i, dof] = 1.
            
        # Physical, each row is M*u    
        for i, ki in enumerate(k, 2):
            u = np.sin(ki*pi*xj)
            # Normalize eigenvector u. M*u = 1
            u /= sqrt((2. + cos(ki*pi/n))/6.)
            # We want M*u for the norm. Since u is eigenvector we are
            # left with ...
            u *= h/3.*(2. + cos(ki*pi/n))
            U[i] = u
        warning('GEVP setup took %g s' % time.stop())

        lmbda = np.r_[1., 1., lmbda0]

        self.W = U.T
        self.lmbda = lmbda

    def get_s_norm(self, s, as_type):
        B = np.dot(self.W * self.lmbda**s, self.W.T)
        
        mat = PETSc.Mat().createDense(size=len(B), array=B)
        return PETScMatrix(mat)

    
def generator(mesh, InterpolationNorm, get_k):
    '''
    Solve Ax = b where A is the eigenvalue representation of (-Delta)^s
    '''
    V = FunctionSpace(mesh, 'CG', 1)
    bdry = DomainBoundary()
    
    bcs = DirichletBC(V, Constant(0), bdry)
    As = InterpolationNorm(V, bcs=bcs)
  
    bp_params = {'k': get_k,
                 'solver': 'cholesky'}

    if isinstance(As, H10_FAST):
        from block.algebraic.petsc import Cholesky
        from scipy.sparse import diags
        from petsc4py import PETSc
        
        class Foo(BP_Operator_Base):

            def compute_k(self, fractionality, space_dim, mesh_size):
                return get_k(fractionality, space_dim, mesh_size)

            
            def shifted_operator(self, shift):
                n = self.V.dim()
                h = self.V.mesh().hmin()
                
                main = np.r_[1., (2./h + shift*4*h/6.)*np.ones(V.dim()-2), 1.]
                off = np.r_[0, (-1./h + shift*h/6.)*np.ones(V.dim()-3), 0.]

                A = diags([off, main, off], [-1, 0, 1]).tocsr()
                A = PETSc.Mat().createAIJ(size=A.shape,
                                          csr=(A.indptr, A.indices, A.data)) 
                return PETScMatrix(A)

            
            def solve_shifted_problem(self, operator, x, vector):
                Ainv = Cholesky(operator)
                x[:] = Ainv*vector
                return 1, x
            
        make_B = lambda s, V=V: Foo(s, V) 
        
        # Need to define here how to get shifted operators from A, M
    else:
        # The norm is constructed from V
        make_B = lambda s: BP_H10Norm(V, bcs, s, bp_params)

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
        bcs.apply(x)
    
        # Rhs
        v = TestFunction(V)
        b = assemble(inner(f, v)*dx)
        bcs.apply(b)
    
        Ainv = ConjGrad(A, precond=B, initial_guess=x, tolerance=1e-13, maxiter=500, show=2)

        # Compute solution
        x = Ainv * b

        nsolves, niters_per_solve = B.nsolves, float(B.niters)/B.nsolves
        k = get_k(s, V.dim(), mesh.hmin())
        print 'with k = %g, solves and niterations per solve[%d(%.4f)]' % (k, nsolves, niters_per_solve)

        
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

    parser = argparse.ArgumentParser()
    
    parser.add_argument('s', help='Exponent of the operator or start:step:end')
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                         default='')
    parser.add_argument('-fgevp', type=int, help='Use closed form eigenvalue solver to compute H^s norm (only D1)',
                        default=0)
    args = parser.parse_args()

    # Fractionality series
    try:
        s_values = [float(args.s)]
    except ValueError:
        # Parse linspace
        start, step, end = map(float, re.compile(r'[+-]?\d+(?:\.\d+)?').findall(args.s))
        # FIXME: map needed bacause of silly types in fenicsii
        s_values = map(float, np.arange(start, end+step, step))

    # Tweak this
    get_k = lambda s, N, h: 5.0*1./ln(N)
    
    InterpolationNorm = H10_FAST if args.fgevp else H10_L2_InterpolationNorm

    results = defaultdict(list)
    sizes = []
    for level, n in enumerate([2**i for i in range(5, 5+args.n)], 1):
        print '\t\t\033[1;37;31m%s\033[0m' % ('level %d, size %d' % (level, n+1))

        mesh = UnitIntervalMesh(n)
        gen = generator(mesh, InterpolationNorm, get_k)

        for s in s_values:
            size, niters, cond = compute(gen, s=s)

            msg = '@s = %g, Current iters %d, cond %.2f, previous %r'
            print '\033[1;37;34m%s\033[0m' % (msg % (s, niters, cond, results[s][::-1]))
            results[s].append((niters, cond))
        sizes.append((size, ))

        # Log after each n in case of kill signal for large n
        # Change make header aware properly
        if args.log:
            log_results(args, sizes, results, name=basename(__file__), fmt='%d %d %g')

    plt.legend(loc='best')
    plt.show()
