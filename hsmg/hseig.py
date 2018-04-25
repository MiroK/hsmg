from block.block_base import block_base
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from petsc4py import PETSc
import numpy as np
from dolfin import *


class InterpolationMatrix(block_base):
    '''
    Given spd matrices A, M this matrix is (M*U)*Lambda^s(M*U)' where 
    A*U = M*U*Lambda and U'*M*U = I
    '''
    def __init__(self, A, M, s, tol=1E-10):
        assert between(s, (-1, 1))
        # Verify symmetry
        assert as_backend_type(A).mat().isHermitian(tol)
        assert as_backend_type(M).mat().isHermitian(tol)

        self.lmbda = None
        self.U = None
        self.matrix = None

        self.A = A
        self.M = M

        self.s = s
        
    def matvec(self, b):
        if self.matrix is None:
            M = self.M.array()
            self.lmbda, self.U = eigh(self.A.array(), M)
            # Build the matrix representation
            W = M.dot(self.U)
            array = csr_matrix((W.dot(np.diag(self.lmbda**self.s))).dot(W.T))
            A = PETSc.Mat().createAIJ(size=array.shape,
                                      csr=(array.indptr, array.indices, array.data))
            self.matrix = PETScMatrix(A)
        return self.matrix*b

    def __pow__(self, power):
        '''A**-1 computes the inverse w/out recomputiong the eigenvalues'''
        assert isinstance(power, int)
        # NOTE: that we return a PETScMatrix not interpolation norm.
        # For the returdned object the **-1 is no longer defined - cbc
        # block allows only positve powers
        if power == -1:
            if self.lmbda is None:
                self.lmbda, self.U = eigh(self.A.array(), self.M.array())
                
            W = self.U
            array = csr_matrix((W.dot(np.diag(self.lmbda**(-self.s)))).dot(W.T))
            A = PETSc.Mat().createAIJ(size=array.shape,
                                  csr=(array.indptr, array.indices, array.data))
            return PETScMatrix(A)
        # Fallback to cbc.block
        else:
            return block_base.__pow__(self, power)


def HsNorm(V, s, bcs=None):
    '''Is based on powers of -Delta + u'''
    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':

        h = CellSize(V.mesh())
        h_avg = avg(h)
        n = FacetNormal(V.mesh())
        # NOTE: most of these terms vanish for DG zero. This is SIP
        # formulation with all the penalty constants equal to 1
        # FIXME: bcs here
        a = inner(grad(v), grad(u))*dx \
            - inner(avg(grad(v)), jump(u, n))*dS \
            - inner(jump(v, n), avg(grad(u)))*dS \
            + h_avg**(-1)*inner(jump(v, n), jump(u, n))*dS \
            - inner(grad(v), u*n)*ds \
            - inner(v*n, grad(u))*ds \
            + h**(-1)*inner(v, u)*ds
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
