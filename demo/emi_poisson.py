from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.convert import block_diagonal_matrix

from block import block_mat, block_assemble
from block.algebraic.petsc import LU, InvDiag, AMG

from dolfin import *


iface = CompiledSubDomain(iface_code)


# FIXME: merge mesh from subdomains to fenics_ii-dev
#        the form for DG0 element in DG0 
#
#        babuska
#        EMI
#        cleanup mortar to be nice for assembly  

def demo(n, beta=1E-9)
    '''
    =================
    |               |
    |   [ i ]       |
    |            e  |
    =================
    
    We solve: Find sigma \in [H(div, Omega)]^2, tau \in L2(Omega), 
    p in H0.5(Gamma) such that 

    (kappa^-1 sigma, tau) - (u, div tau) + (p, tau . n) = 0
    -(div sigma, v)       + (u, v)                      = 0
    (sigma . n, q)                       -beta(p, q)    = (g, q)

    To have the Schur complement dominated by fractional term we let
    beta very small (correspond to very large time steps in the EMI 
    problem 
    '''
    kappe_e = Constant(1)
    kappa_i = Constant(1.5)
    beta = Constant(beta)
    q = Expression('sin(k*pi*(x[0]+x[1]))', k=3, degree=3)

    # ---------------------
    
    omega = UnitSquareMesh(4*n, 4*n)
    facet_f = FacetFunction('size_t', omega, 0)
    interface.mark(facet_f, 1)

    gamma = EmbeddedMesh(omega, facet_f, 1, normal=[0.0, 0.0])

    S = FunctionSpace(omega, 'RT', 1)        # sigma
    V = FunctionSpace(omega, 'DG', 0)        # u
    Q = FunctionSpace(gamma.mesh, 'DG', 0)   # p
    W = [S, V, Q]

    sigma, u, p = map(TrialFunction, W)
    tau, v, q = map(TestFunction, W)

    dX = Measure('dx', domain=omega, subdomain_data=subdomains)
    dxGamma = dx(domain=gamma.mesh)
        
    n_gamma = gamma.normal('+')          # Outer of inner square
    
    # System - for symmetry
    a00 = inner(Constant(1./kappa_e)*sigma, tau)*dX(0) +\
          inner(Constant(1./kappa_i)*sigma, tau)*dX(1)
    a01 = -inner(u, div(tau))*dX
    a02 = inner(dot(tau('+'), n_gamma), p)*dxGamma

    a10 = -inner(div(sigma), v)*dX
    a11 = inner(u, v)*dX

    a20 = inner(dot(sigma('+'), n_gamma), q)*dxGamma
    a22 = -Constant(1./beta)*inner(p, q)*dxGamma   

    A00, A01, A10, A11, A22 = map(assemble, (a00, a01, a10, a11, a22))
    A02 = trace_assemble(a02, gamma)
    A20 = trace_assemble(a20, gamma)

    AA = block_mat([[A00, A01, A02],
                    [A10, A11, A12],
                    [A20, A21, A22]])

    bb = block_assemble([inner(Constant((0, 0)), tau)*dx,
                         inner(Constant(0), v)*dx,
                         inner(g, q)*dxGamma])

    # Block of Riesz preconditioner
    B00 = LU(assemble(inner(sigma, tau)*dX + inner(div(sigma), div(tau))*dX))
    B11 = InvDiag(inner(u, v)*dX)

        # These are terms needed for the alternate formulation in
        # Broken H1, H1, L2
        dmyS = Measure('dS', domain=omega, subdomain_data=interface)
        n = FacetNormal(omega)
    
        # L2
        # NOTE: Excluding surface term -> problem
        q00 = inner(sigma, tau)*dX + \
              inner(dot(sigma('+'), n('+')), dot(tau('+'), n('+')))*dmyS(1)
        Q00 = assemble(q00)
    
        # H1
        h = CellSize(omega)
        h_avg = avg(h)
        # Avoid the interface, we have jump in u -> can't have global H1
        # NOTE: including whole skeleton -> problem :)
        q11 = h_avg**(-1)*dot(jump(v, n), jump(u, n))*dmyS(0) +\
              h**(-1)*dot(u, v)*ds+\
              inner(u, v)*dX
        Q11 = assemble(q11)

        # Simple mass matrix (diagonal with DG0)
        # NOTE: does not like fractions :)
        q22 = Constant(1./beta)*inner(p, q)*dx
        Q22 = assemble(q22)
    
        # Remember
        self.AA = [[A00, A01, A02],
                   [A10, A11,   0],
                   [A20,   0, A22]]

        # For rhs assembly
        self.dX = dX
        # Preconditioner
        self.P = map(assemble, (p00, p11))
        self.beta = beta
        # Theoretical value is 0.5 but since Schur is 1./dt*M + H(0.5)
        # M might dominate the spectrum and then you don't see fractional
        # term. So I keep it to play with it
        self.s_value = params.get('s_value', 0.5)

        # Spaces
        self.W = W

        # Block of no_frac preconditioner
        self.Q = [Q00, Q11, Q22]
                

    def __no_frac_preconditioner(self, riesz_map):
        '''Riesz map for iters. Try to bypass fractional norms'''
        B00, B11, B22 = self.Q
        
        if not riesz_map:
            BB = block_diagonal_matrix([B00, B11, B22])
        else:
            BB = block_diagonal_matrix([AMG(B00), AMG(B11), InvDiag(B22)])
        return BB

    def __frac_preconditioner(self, riesz_map):
        '''
        The precondiioner is based on 
            H1(div) x L2(Omega) x (1./dt*L2(gamma) \cap H^0.5(gamma))
        '''        
        B00 = self.P[0]
        B11 = self.P[1]
        
        X = H1_L2_InterpolationNorm(self.W[-1])
        # X = DG0_H1_L2_InterpolationNorm(self.W[-1])
        
        if not riesz_map:
            Hnorm = X.get_s_norm(s=[(1./self.beta(0), 0), (1.0, self.s_value)],
                                 as_type=PETScMatrix)
 
            BB = block_diagonal_matrix([B00, B11, Hnorm])
        else:
            B00 = LU(B00)
            B11 = InvDiag(B11)
            Hnorm_inv = X.get_s_norm_inv(s=[(1./self.beta(0), 0), (1.0, self.s_value)],
                                         as_type=PETScMatrix)
                                         
            BB = block_diagonal_matrix([B00, B11, Hnorm_inv])
return BB

from dolfin import as_backend_type
import os, sys


def krylov_solve(A, b, B, tol):
    '''Start from random init guess'''
    x = A.create_vec()
    [as_backend_type(xi).vec().setRandom() for xi in x]
    Ainv = MinRes(A, precond=B, show=2, tolerance=tol, initial_guess=x, maxiter=250)
    x = Ainv*b
    
return len(Ainv.residuals) - 1
