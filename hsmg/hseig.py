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
    ones = A.create_vec()
    ones.set_local(np.ones_like(ones.local_size()))
    
    d = A*ones
    
    if s is None:
        return d
    # Scaling
    return PETSc.Vec().createWithArray(d.get_local()**s)

    
class InterpolationMatrix(block_base):
    '''
    Given spd matrices A, M this operator is (M*U)*Lambda^s(M*U)' where 
    A*U = M*U*Lambda and U'*M*U = I
    '''
    def __init__(self, A, M, s, tol=1E-10, lump='', use_pinv=False):
        # Verify symmetry
        assert as_backend_type(A).mat().isHermitian(tol)
        assert as_backend_type(M).mat().isHermitian(tol)
        # This is a layzy thing so the operator is not formed until
        # its action is needed
        self.lmbda = None
        self.U = None
        self.matrix = None

        self.A = A
        self.M = M

        # The s doesn't have to be a single power; we allos for alpha*()**s_alpha
        # NOTE: this is a bit dangerous with DirichletBcs as it rescles
        # the "zeroed" rows. 
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
        self.use_pinv = use_pinv
        
    def create_vec(self, dim=0):
        return self.A.create_vec(dim)
        
    def matvec(self, b):
        '''Action on b vector'''
        # We have the matrix
        if self.matrix is not None:
            return self.matrix*b

        # We compute it for the first time
        M = self.M.array()
        if self.lmbda is None and self.U is None:
            info('Computing %d eigenvalues for InterpolationMatrix' % b.size())
            t = Timer('eigh')
            # Solve as generalized eigenvalue problem
            if not self.lump:
                M = self.M.array()
                self.lmbda, self.U = my_eigh(self.A.array(), M)
                info('Done %g' % t.stop())
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

        assert self.use_pinv or all(self.lmbda > 0)  # pos def

        # Build the matrix representation
        W = M.dot(self.U)

        # Build the diagonal
        diag = np.zeros_like(self.lmbda)

        # Zero eigenvalue is 1E-12
        idx = np.abs(self.lmbda) < 1E-12*len(W)
        self.lmbda[idx] = 1
        
        for weight, exponent in self.s:
            diag = diag + weight*self.lmbda**exponent

        array = csr_matrix((W.dot(np.diag(diag))).dot(W.T))
                
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


def Hs0Norm(V, s, bcs, kappa=Constant(1), lump='', use_pinv=False):
    '''
    Interpolation matrix with A based on -Delta and M based on I.

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
      bcs = DirichletBC instance specifying boundary conditions
            for DG spaces FacetFunction which marks domain of homog.
            Dirichlet as 1. Alternatively True means 'on_boundary'

    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = kappa*inner(u, v)*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        # assert V.ufl_element().degree() == 0
        # FIXME: kappa \neq 1
        mesh = V.mesh()
        h = CellDiameter(mesh)
        h_avg = avg(h)
        n = FacetNormal(mesh)

        # Some heuristic to get the 
        penalty = {0: 1, 1: 2, 2: 8, 3: 16}[V.ufl_element().degree()]
        penalty *= mesh.topology().dim()

        alpha = Constant(penalty)
        gamma = Constant(penalty)
        # NOTE: weakly include bcs
        # a = h_avg**(-1)*dot(jump(v), jump(u))*dS
        a = (kappa*dot(grad(v), grad(u))*dx 
            - kappa*dot(avg(grad(v)), jump(u, n))*dS 
            - kappa*dot(jump(v, n), avg(grad(u)))*dS 
            + kappa*alpha/h_avg*dot(jump(v), jump(u))*dS)

        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        # Whole boundary
        if isinstance(bcs, bool):
            bcs and DomainBoundary().mark(facet_f, 1)
        # Where they want it
        else:
            facet_f = bcs
        dBdry = ds(domain=mesh, subdomain_data=facet_f, subdomain_id=1)
            
        a += (-kappa*dot(grad(v), u*n)*dBdry
              - kappa*dot(v*n, grad(u))*dBdry
              + kappa*(gamma/h)*v*u*dBdry)        
            
        A, M = map(assemble, (a, m))
    else:
        a = inner(kappa*grad(u), grad(v))*dx

        A, _ = assemble_system(a, L, bcs)
        M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s, lump=lump, use_pinv=use_pinv)


def HsNorm(V, s, bcs=None, kappa=Constant(1), lump='', use_pinv=False):
    '''
    Interpolation matrix with A based on -Delta and M based on I.

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
      bcs = DirichletBC instance specifying boundary conditions
            for DG spaces FacetFunction which marks domain of homog.
            Dirichlet as 1. Alternatively True means 'on_boundary'

    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = kappa*inner(u, v)*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        # assert V.ufl_element().degree() == 0
        # FIXME: kappa \neq 1
        mesh = V.mesh()
        h = CellDiameter(mesh)
        h_avg = avg(h)
        n = FacetNormal(mesh)

        # Some heuristic to get the 
        penalty = {0: 1, 1: 2, 2: 8, 3: 16}[V.ufl_element().degree()]
        penalty *= mesh.topology().dim()

        alpha = Constant(penalty)
        gamma = Constant(penalty)
        # NOTE: weakly include bcs
        # a = h_avg**(-1)*dot(jump(v), jump(u))*dS
        a = (kappa*dot(grad(v), grad(u))*dx 
            - kappa*dot(avg(grad(v)), jump(u, n))*dS 
            - kappa*dot(jump(v, n), avg(grad(u)))*dS 
            + kappa*alpha/h_avg*dot(jump(v), jump(u))*dS)

        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        # Whole boundary
        if isinstance(bcs, bool):
            bcs and DomainBoundary().mark(facet_f, 1)
        # Where they want it
        else:
            facet_f = bcs
        dBdry = ds(domain=mesh, subdomain_data=facet_f, subdomain_id=1)
            
        a += (-kappa*dot(grad(v), u*n)*dBdry
              - kappa*dot(v*n, grad(u))*dBdry
              + kappa*(gamma/h)*v*u*dBdry)        
            
        A, M = map(assemble, (a+m, m))
    else:
        a = inner(kappa*grad(u), grad(v))*dx

        A, _ = assemble_system(a+m, L, bcs)
        M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s, lump=lump, use_pinv=use_pinv)


# Backward compatibility
wHsNorm = HsNorm
wHs0Norm = Hs0Norm


# --------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import *
    import numpy as np
    import scipy.linalg as spla

    if False:
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


    import block

    s = -0.123
    mu = 1E4
    
    mesh = UnitIntervalMesh(128)

    V = FunctionSpace(mesh, 'CG', 1)
    x = Function(V).vector()
    x.set_local(np.random.rand(x.local_size()))
    
    A0 = HsNorm(V, s=s, bcs=None)
    y0 = mu*A0*x

    A = wHsNorm(V, s=s, bcs=None, kappa=Constant(mu))
    y = A*x

    print (y - y0).norm('l2') 

    # print (y-y0).get_local()

    V = FunctionSpace(mesh, 'DG', 0)
    x = Function(V).vector()
    x.set_local(np.random.rand(x.local_size()))
    
    A0 = HsNorm(V, s=s, bcs=None)
    y0 = mu*A0*x

    A = wHsNorm(V, s=s, bcs=None, kappa=Constant(mu))
    y = A*x

    print (y - y0).norm('l2') 
