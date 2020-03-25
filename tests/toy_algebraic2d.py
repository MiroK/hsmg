from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from petsc4py import PETSc
from dolfin import *
import numpy as np

import pyamg


def Hs_gen(n):
    '''Fractional Laplacian'''
    assert n > 0
    assert -1 <= 0 <= 1
    mesh = UnitSquareMesh(*(n, )*2)
    V = FunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    m = inner(u, v)*dx
    
    A, M = map(assemble, (a, m))

    A_, M_ = A.array(), M.array()
    print('GEVP %d x %d' % (V.dim(), V.dim()))
    timer = Timer('Hs')
    lmbda, U = eigh(A_, M_)
    print('Done %s' % timer.stop())
    
    W = M_.dot(U)

    return lambda s, W=W, lmbda=lmbda:(V, csr_matrix(W.dot(np.diag(lmbda**s).dot(W.T))))


def csr_to_dolfin(A):
    '''PETScMatrix form csr_matrix'''
    Amat = PETSc.Mat().createAIJ(size=A.shape,
                                 csr=(A.indptr, A.indices, A.data))
    Amat = PETScMatrix(Amat)

    return Amat

# --------------------------------------------------------------------

if __name__ == '__main__':
    from block.iterative import ConjGrad
    from hsmg.hs_mg import HsNormMG, HsNormAMG
    import matplotlib.pyplot as plt
    import pyamg

    n = 2**6
    # This is a mapping from s to the CSR-formatted matrix representation
    # of fractional laplacian (-Delta + I)^s
    Hs = Hs_gen(n)

    frac_preconditioner = 'amg'
    bdry = None
    s = 0.75    
    if frac_preconditioner == 'mg':
        mg_params = {'macro_size': 1,
                     'nlevels': 4,
                     'eta': 0.4}
        precond = HsNormMG
    else:
        assert frac_preconditioner == 'amg'
        # Pass a fully configured solver that is constructed once A
        # goes in
        mg_params = {'pyamg_solver': lambda A: pyamg.ruge_stuben_solver(A)}
        precond = HsNormAMG
        
    V, A = Hs(s)  # Csr    
    Amat = csr_to_dolfin(A)  # Compat with cbc.block
    
    precond = precond(V, bdry=None, s=s, mg_params=mg_params)
    # Solve it by preconditioned CG
    HsInvMg = ConjGrad(Amat, precond=precond, tolerance=1E-10, show=3)

    vec = Vector(mpi_comm_world(), V.dim())
    vec.set_local(np.random.rand(V.dim()))
    x = HsInvMg*vec
    print V.dim()
    
    # How AMG solves stuff if it has access to the matrix
    if False:
        
        b = np.random.rand(n+1)
        s_residuals = {}
        for s in [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
            _, A = Hs(s)
            ml = pyamg.ruge_stuben_solver(A) 
            print(ml)  # Hierarchy information
            
            residuals = []
            x = ml.solve(b, tol=1e-10, residuals=residuals)  
            print('%s -> %s residual: ' % (s, len(residuals)), np.linalg.norm(b-A*x))
            
            s_residuals[s] = residuals


        plt.figure()
        for s in sorted(s_residuals.keys()):
            plt.semilogy(s_residuals[s], label=str(s))
        plt.legend()
        plt.show()
