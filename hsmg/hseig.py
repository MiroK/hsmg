from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from petsc4py import PETSc
import numpy as np
from dolfin import *


class InterpolationMatrix(PETScMatrix):
    '''
    Given spd matrices A, M this matrix is (M*U)*Lambda^s(M*U)' where 
    A*U = M*U*Lambda and U'*M*U = I
    '''
    def __init__(self, A, M, s, tol=1E-10):
        assert between(s, (-1, 1))
        # Verify symmetry
        assert as_backend_type(A).mat().isHermitian(tol)
        assert as_backend_type(M).mat().isHermitian(tol)
        # FIXME: handle singular (Laplacian w/out Dirichlet bcs)
        eigw, eigv = eigh(A.array(), M.array())
        assert all(w > 0 for w in eigw)

        self.lmbda = eigw
        self.U = eigv
        self.M = M.array()
        self.s = s

        # Build the matrix representation
        W = self.M.dot(self.U)
        array = csr_matrix((W.dot(np.diag(self.lmbda**self.s))).dot(W.T))
        A = PETSc.Mat().createAIJ(size=array.shape,
                                  csr=(array.indptr, array.indices, array.data))
        # Yes that' me
        PETScMatrix.__init__(self, A)

    def __pow__(self, power):
        '''A**-1 computes the inverse w/out recomputiong the eigenvalues'''
        assert isinstance(power, int)
        # NOTE: that we return a PETScMatrix not interpolation norm.
        # For the returdned object the **-1 is no longer defined - cbc
        # block allows only positve powers
        if power == -1:
            W = self.U
            array = csr_matrix((W.dot(np.diag(self.lmbda**(-self.s)))).dot(W.T))
            A = PETSc.Mat().createAIJ(size=array.shape,
                                  csr=(array.indptr, array.indices, array.data))
            return PETScMatrix(A)
        # Fallback to cbc.block
        else:
            assert power >= 0
            if power == 0:
                return 1
            else:
                return self*(self**(power-1))


def HsNorm(V, s, bcs=None):
    '''Is based on powers of -Delta + u'''
    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        assert V.ufl_element().degree() == 0

        h = CellSize(V.mesh())
        h_avg = avg(h)
        # FIXME: bcs here
        a = h_avg**(-1)*dot(jump(v), jump(u))*dS + h**(-1)*dot(u, v)*ds
    else:
        a = inner(grad(u), grad(v))*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx

    A, _ = assemble_system(a+m, L, bcs)
    M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s)


def Hs0Norm(V, s, bcs):
    '''Is based on powers of -Delta'''
    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        assert V.ufl_element().degree() == 0

        h = CellSize(V.mesh())
        h_avg = avg(h)
        # FIXME: bcs here
        a = h_avg**(-1)*dot(jump(v), jump(u))*dS + h**(-1)*dot(u, v)*ds
    else:
        a = inner(grad(u), grad(v))*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx

    A, _ = assemble_system(a, L, bcs)
    M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s)
