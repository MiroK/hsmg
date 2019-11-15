from block.block_base import block_base
from scipy.sparse import csr_matrix, diags
from utils import my_eigh
from petsc4py import PETSc
import numpy.ma as mask
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
    def __init__(self, A, M, s, tol=1E-10, lump='', cap_zero_mean=False):
        # Verify symmetry
        assert as_backend_type(A).mat().isHermitian(tol)
        assert as_backend_type(M).mat().isHermitian(tol)

        self.lmbda = None
        self.U = None
        self.matrix = None

        self.A = A
        self.M = M

        # Allow tuples being combinations (weight, exponents)
        if isinstance(s, (int, float)):
            s = (1, s)
        if isinstance(s, tuple):
            s = [s]
            
        assert isinstance(s, list)
        assert [-1-tol < exponent < 1+tol for _, exponent in s]
        
        self.s = s
        # How to take M into account
        self.lump = lump
        self.cap_zero_mean = cap_zero_mean
        
    def create_vec(self, dim=0):
        return self.A.create_vec(dim)
        
    def matvec(self, b):
        '''Action on b vector'''
        if self.matrix is None:
            M = self.M.array()
            if self.lmbda is None and self.U is None:
                info('Computing %d eigenvalues for InterpolationMatrix' % b.size())
                t = Timer('eigh')
                # Solve as generalized eigenvalue problem
                if not self.lump:
                    M = self.M.array()
                    self.lmbda, self.U = my_eigh(self.A.array(), M)
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
                info('Done %g' % t.stop())
            assert self.cap_zero_mean or all(self.lmbda > 0)  # pos def

            # Build the matrix representation
            W = M.dot(self.U)

            # Build the diagonal
            diag = np.zeros_like(self.lmbda)

            # Zero eigenvalue is 1E-12
            idx = np.abs(self.lmbda) < 1E-12*len(W)
            print np.any(idx), 1E-12*len(W)
            self.lmbda = mask.masked_array(self.lmbda, idx, fill_value=0.)
        
            for weight, exponent in self.s:
                diag = diag + weight*self.lmbda**exponent

            if len(idx):
                Z = W[:, idx]
                array = csr_matrix((W.dot(np.diag(diag.data))).dot(W.T) + Z.dot(Z.T))
            else:    
                array = csr_matrix((W.dot(np.diag(diag.data))).dot(W.T))
                
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
            if self.cap_zero_mean: raise ValueError
            
            if self.lmbda is None:
                info('Computing %d eigenvalues for InterpolationMatrix' % self.M.size(0))
                self.lmbda, self.U = my_eigh(self.A.array(), self.M.array())
                self.lmbda = mask.masked_array(self.lmbda, np.abs(self.lmbda) < 1E-12, fill_value=0.)
                
            W = self.U
            
            diag = np.zeros_like(self.lmbda)
            for weight, exponent in self.s:
                diag = diag + weight*self.lmbda**exponent
            diag = diag**-1.
            
            array = csr_matrix((W.dot(np.diag(diag.data))).dot(W.T))
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
    mesh = V.mesh()
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        # FIXME: kappa \neq 1
        h = CellDiameter(mesh)
        h_avg = avg(h)

        a = h_avg**(-1)*dot(jump(v), jump(u))*dS + inner(u, v)*dx
        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        # No bcs
        if bcs is None: bcs = False
        # Whole boundary
        if isinstance(bcs, bool):
            bcs and DomainBoundary().mark(facet_f, 1)
        # Where they want it
        else:
            facet_f = bcs
        # Add it
        a += h**(-1)*inner(u, v)*ds(domain=mesh, subdomain_data=facet_f, subdomain_id=1)            

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
        mesh = V.mesh()
        h = CellDiameter(mesh)
        h_avg = avg(h)
        # NOTE: weakly include bcs
        a = h_avg**(-1)*dot(jump(v), jump(u))*dS

        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        # Whole boundary
        if isinstance(bcs, bool):
            bcs and DomainBoundary().mark(facet_f, 1)
        # Where they want it
        else:
            facet_f = bcs
        a += h**(-1)*dot(u, v)*ds(domain=mesh, subdomain_data=facet_f, subdomain_id=1)

        A, M = map(assemble, (a, m))
    else:
        a = inner(kappa*grad(u), grad(v))*dx

        A, _ = assemble_system(a, L, bcs)
        M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s, lump=lump)


def HsZNorm(V, s, kappa=Constant(1)):
    '''
    Hs \cap L^2_0

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx

    if V.ufl_element().family() == 'Discontinuous Lagrange':
        assert V.ufl_element().degree() == 0
        # FIXME: kappa \neq 1
        mesh = V.mesh()
        h = CellDiameter(mesh)
        h_avg = avg(h)

        a = h_avg**(-1)*dot(jump(v), jump(u))*dS
    else:
        a = inner(kappa*grad(u), grad(v))*dx
        
    A, M = map(assemble, (a, m))
    
    return InterpolationMatrix(A, M, s, cap_zero_mean=True)



def L20Norm(V):
    '''Matrix for inner product of L^2_0 discretized by V'''
    assert V.ufl_element().value_shape() == ()
    one = interpolate(Constant(1), V)
    # Make sure we have (1, 1) = 1
    one.vector()[:] *= 1./sqrt(assemble(inner(one, one)*dx))

    u, v = TrialFunction(V), TestFunction(V)
    m = inner(u, v)*dx
    M = assemble(m)

    z = M*one.vector()
    # Based on (Pu, Pv) where P is the projector
    M = M.array() - np.outer(z.get_local(), z.get_local())
    M = csr_matrix(M)
    M = PETSc.Mat().createAIJ(size=M.shape,
                              csr=(M.indptr, M.indices, M.data))

    return PETScMatrix(M)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    import numpy as np
    import scipy.linalg as spla
    
    mesh = UnitSquareMesh(16, 16)
    V = FunctionSpace(mesh, 'CG', 1)
    x = Function(V).vector()

    y = interpolate(Constant(2), V).vector()
    A = HsZNorm(V, s=-0.5)

    A = HsNorm(V, s=-0.5)
    A*x

    AA = HsNorm(V, s=[(1, -0.5), (2, -0.25)])
    B = AA**-1

    print (x - B*AA*x).norm('l2'), '<<<<'
    

    B = HsNorm(V, s=-0.5, lump='row')
    B*x

    print np.linalg.norm(A.matrix.array() - B.matrix.array())

    print 
    z = interpolate(Constant(2), V).vector()
    M = L20Norm(V)
    print (M*z).norm('l2')

    x = interpolate(Expression('sin(pi*(x[0]+x[1]))', degree=3), V)
    mine = x.vector().inner(M*x.vector())
    true = assemble(inner(x, x)*dx)

    print abs(mine-true)
