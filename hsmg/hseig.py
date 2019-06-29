from block.block_base import block_base
from scipy.sparse import csr_matrix, diags
from scipy.linalg import eigh
from petsc4py import PETSc
import numpy as np
from dolfin import *


def Diag(A, s=None):
    '''A diagonal of matrix as PETSc.Vec'''
    if s is None:
        return as_backend_type(A).mat().getDiagonal()
    # Scale
    d = as_backend_type(A).mat().getDiagonal()
    d.setValues(d.array_r**s)
    
    return d

    
def LumpedDiag(A, s=None):
    '''Row sum of A as PETSc.Vec'''
    mat = as_backend_type(A).mat()
    d = map(lambda i: np.linalg.norm(mat.getRow(i)[1], 1), range(A.size(0)))
    
    if s is None:
        return PETSc.Vec().createWithArray(np.array(d))
    # Scaling
    return PETSc.Vec().createWithArray(np.array(d)**s)

    
class InterpolationMatrix(block_base):
    '''
    Given spd matrices A, M this operator is (M*U)*Lambda^s(M*U)' where 
    A*U = M*U*Lambda and U'*M*U = I
    '''
    def __init__(self, A, M, s, tol=1E-10, lump=''):
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
        # How to take M into account
        self.lump = lump
        
    def create_vec(self, dim=0):
        return self.A.create_vec(dim)
        
    def matvec(self, b):
        '''Action on b vector'''
        if self.matrix is None:
            
            info('Computing %d eigenvalues for InterpolationMatrix' % b.size())
            t = Timer('eigh')
            # Solve as generalized eigenvalue problem
            if not self.lump:
                M = self.M.array()
                self.lmbda, self.U = eigh(self.A.array(), M)
            else:
                # The idea here is A u = l M u
                # Let u = M-0.5 v so then M-0.5 A M-0.5 v = l v
                # Solve EVP for          <------------>
                # Eigenvector need M-0.5*v
                if self.lump == 'diag':
                    d = Diag(self.M, -0.5)
                    M = Diag(self.M)
                else:
                    d = LumpedDiag(self.M, -0.5)
                    M = LumpedDiag(self.M)
                # Using only the approx of mass matrix
                M = diags(M.array)
                
                # Eigenvalues
                Amat = as_backend_type(self.A).mat()
                Amat.diagonalScale(d, d)  # Build M-0.5 A M-0.5
                self.lmbda, V = np.linalg.eigh(PETScMatrix(Amat).array())
                # Map eigenvectors
                self.U = diags(d.array).dot(V)
                
            assert all(self.lmbda > 0)  # pos def
            info('Done %g' % t.stop())
            
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
                info('Computing %d eigenvalues for InterpolationMatrix' % self.M.size(0))
                self.lmbda, self.U = eigh(self.A.array(), self.M.array())
                
            W = self.U
            array = csr_matrix((W.dot(np.diag(self.lmbda**(-self.s)))).dot(W.T))
            A = PETSc.Mat().createAIJ(size=array.shape,
                                  csr=(array.indptr, array.indices, array.data))
            return PETScMatrix(A)
        # Fallback to cbc.block
        else:
            assert power > 0
            return block_base.__pow__(self, power)


def HsNorm(V, s, bcs=None, kappa=Constant(1), lump=''):
    '''
    Interpolation matrix with A based on -Delta + I and M based on I.

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
      bcs = DirichletBC instance specifying boundary conditions

    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        # FIXME: kappa \neq 1
        h = CellDiameter(V.mesh())
        h_avg = avg(h)

        a = h_avg**(-1)*dot(jump(v), jump(u))*dS + inner(u, v)*dx
        if bcs is True:
            a += h**(-1)*inner(u, v)*ds            

        return InterpolationMatrix(assemble(a), assemble(m), s, lump=lump)        
    else:
        a = inner(kappa*grad(u), grad(v))*dx
        
    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx

    A, _ = assemble_system(a+m, L, bcs)
    M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s, lump=lump)


def Hs0Norm(V, s, bcs, kappa=Constant(1), lump=''):
    '''
    Interpolation matrix with A based on -Delta and M based on I.

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
      bcs = DirichletBC instance specifying boundary conditions

    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        assert V.ufl_element().degree() == 0
        # FIXME: kappa \neq 1
        h = CellDiameter(V.mesh())
        h_avg = avg(h)
        # NOTE: weakly include bcs
        a = h_avg**(-1)*dot(jump(v), jump(u))*dS + h**(-1)*dot(u, v)*ds

        A, M = map(assemble, (a, m))
    else:
        a = inner(kappa*grad(u), grad(v))*dx

        A, _ = assemble_system(a, L, bcs)
        M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s, lump=lump)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    import numpy as np
    import scipy.linalg as spla
    
    mesh = UnitSquareMesh(64, 64)
    V = FunctionSpace(mesh, 'CG', 1)
    x = Function(V).vector()

    A = HsNorm(V, s=-0.5)
    A*x


    B = HsNorm(V, s=-0.5, lump='row')
    B*x

    print np.linalg.norm(A.matrix.array() - B.matrix.array())

