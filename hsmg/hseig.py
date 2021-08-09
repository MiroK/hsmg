from block.block_base import block_base
from scipy.sparse import csr_matrix, diags
from scipy.linalg import eigh
from hsmg.utils import my_eigh
from petsc4py import PETSc
import numpy.ma as mask
import numpy as np
from dolfin import *

    
class InterpolationMatrix(block_base):
    '''
    Given spd matrices A, M this operator is (M*U)*Lambda^s(M*U)' where 
    A*U = M*U*Lambda and U'*M*U = I
    '''
    def __init__(self, A, M, s, tol=1E-10):
        # Verify symmetry
        assert as_backend_type(A).mat().isHermitian(tol)
        assert as_backend_type(M).mat().isHermitian(tol)
        # This is a layzy thing so the operator is not formed until
        # its action is needed
        self.lmbda = None
        self.U = None
        self.matrix = None  # Matrix representation of the operator

        self.A = A
        self.M = M

        assert -1-tol < s < 1+tol
        self.s = s
        
    def create_vec(self, dim=0):
        return self.A.create_vec(dim)
        
    def matvec(self, b):
        '''Action on b vector'''
        # We have the matrix
        if self.matrix is not None:
            return self.matrix*b

        self.collapse()
        return self.matrix*b

    def collapse(self):
        '''Compute matrix representation of the operator'''
        if self.matrix is not None:
            return self.matrix
    
        info('Computing %d eigenvalues for InterpolationMatrix' % self.A.size(0))
        t = Timer('eigh')
        # Solve as generalized eigenvalue problem
        try:
            self.lmbda, self.U = my_eigh(self.A.array(), self.M.array())
        except:
            self.lmbda, self.U = eigh(self.A.array(), self.M.array())

        assert np.all(self.lmbda > 0), np.min(np.abs(self.lmbda))  # pos def

        M = self.M.array()
        # Build the matrix representation
        W = M.dot(self.U)

        # Build the diagonal
        diag = np.zeros_like(self.lmbda)

        # Zero eigenvalue is 1E-12
        diag = self.lmbda**self.s

        array = csr_matrix((W.dot(np.diag(diag))).dot(W.T))
                
        A = PETSc.Mat().createAIJ(size=array.shape,
                                  csr=(array.indptr, array.indices, array.data))
        self.matrix = PETScMatrix(A)

        return self.matrix
    
    def __pow__(self, power):
        '''A**-1 computes the inverse w/out recomputiog the eigenvalues'''
        assert isinstance(power, int)
        # NOTE: that we return a PETScMatrix not interpolation norm.
        assert power == -1
        
        if self.matrix is None:
            self.collapse()
                
        W = self.U
        diag = self.lmbda**(-1.*self.s)
            
        array = csr_matrix((W.dot(np.diag(diag.data))).dot(W.T))
        A = PETSc.Mat().createAIJ(size=array.shape,
                                      csr=(array.indptr, array.indices, array.data))
        return PETScMatrix(A)


def Hs0Eig(V, s, bcs, kappa=Constant(1)):
    '''
    Interpolation matrix with A based on -kappa*Delta and M based on kappa*I.

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
      bcs = [(facet function, tag)]

    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = kappa*inner(u, v)*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        mesh = V.mesh()
        h = CellDiameter(mesh)
        h_avg = avg(h)
        n = FacetNormal(mesh)

        # FIXME: heustiristics for SIP
        penalty = {0: 1, 1: 2, 2: 8, 3: 16}[V.ufl_element().degree()]
        penalty *= mesh.topology().dim()

        alpha = Constant(penalty)
        gamma = Constant(penalty)
        # Interior
        a = (kappa*dot(grad(v), grad(u))*dx 
            - kappa*dot(avg(grad(v)), jump(u, n))*dS 
            - kappa*dot(jump(v, n), avg(grad(u)))*dS 
            + kappa*alpha/h_avg*dot(jump(v), jump(u))*dS)

        # Exterior
        for facet_f, tag in bcs:
            assert facet_f.dim() == mesh.topology().dim() - 1

            dBdry = Measure('ds', domain=mesh, subdomain_data=facet_f)
            
            a += (-kappa*dot(grad(v), u*n)*dBdry(tag)
                  - kappa*dot(v*n, grad(u))*dBdry(tag)
                  + kappa*(gamma/h)*v*u*dBdry(tag))

        A, M = map(assemble, (a, m))
    else:
        a = inner(kappa*grad(u), grad(v))*dx
        bcs = [DirichletBC(V, zero, facet_f, tag) for facet_f, tag in bcs]

        A, _ = assemble_system(a, L, bcs)
        M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s)


def HsEig(V, s, bcs=None, kappa=Constant(1)):
    '''
    Interpolation matrix with A based on kappa*(-Delta+I) and M based on kappa*I.

    INPUT:
      V = function space instance
      s = float that is the fractionality exponent
      bcs = [(facet function, tag)]

    OUTPUT:
      InterpolationMatrix
    '''
    u, v = TrialFunction(V), TestFunction(V)
    m = kappa*inner(u, v)*dx

    zero = Constant(np.zeros(v.ufl_element().value_shape()))
    L = inner(zero, v)*dx
    
    if V.ufl_element().family() == 'Discontinuous Lagrange':
        mesh = V.mesh()
        h = CellDiameter(mesh)
        h_avg = avg(h)
        n = FacetNormal(mesh)
        # FIXME: these are just heurstic to get the SIP penalty
        penalty = {0: 1, 1: 2, 2: 8, 3: 16}[V.ufl_element().degree()]
        penalty *= mesh.topology().dim()

        alpha = Constant(penalty)
        gamma = Constant(penalty)

        # Interior
        a = (kappa*dot(grad(v), grad(u))*dx 
            - kappa*dot(avg(grad(v)), jump(u, n))*dS 
            - kappa*dot(jump(v, n), avg(grad(u)))*dS 
            + kappa*alpha/h_avg*dot(jump(v), jump(u))*dS)

        if bcs is not None:
            for facet_f, tag in bcs:
                assert facet_f.dim() == mesh.topology().dim() - 1

                dBdry = Measure('ds', domain=mesh, subdomain_data=facet_f)
            
                a += (-kappa*dot(grad(v), u*n)*dBdry(tag)
                      - kappa*dot(v*n, grad(u))*dBdry(tag)
                      + kappa*(gamma/h)*v*u*dBdry(tag))
            
        A, M = map(assemble, (a+m, m))
    else:
        a = inner(kappa*grad(u), grad(v))*dx
        # Expend
        if bcs is not None:
            bcs = [DirichletBC(V, zero, facet_f, tag) for facet_f, tag in bcs]            
        
        A, _ = assemble_system(a+m, L, bcs)
        M, _ = assemble_system(m, L, bcs)

    return InterpolationMatrix(A, M, s)
