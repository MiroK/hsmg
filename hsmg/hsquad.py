from dolfin import LinearOperator, PETScPreconditioner, PETScKrylovSolver
from dolfin import SubDomain, CompiledSubDomain, between, Constant
from dolfin import DirichletBC, inner, grad, dx, assemble_system
from dolfin import CellSize, avg, dot, jump, dS, ds, zero
from dolfin import TrialFunction, TestFunction
from dolfin import Vector, MPI, solve
import dolfin as df

from block.object_pool import vec_pool
from block.block_base import block_base
from block.algebraic.petsc import Cholesky

from math import sin, pi, sqrt, exp, ceil
from types import FunctionType
import numpy as np


df.set_log_level(df.WARNING)


class BP_Operator_Base(block_base):
    '''
    This is an interface for solvers of equations of the type L ^ s u = b.
    Solution u is computed as u = L^{-s} b where action of L^{-s} is computed
    from its integral representation. This requires solves of the shifted
    problem (L + p*I) y = c.

    One must provide (p -> L + p*I), the solver for it and function which
    controls the number of quadrature points
    '''
    def __init__(self, s, V):
        '''
        I want to solve L^s x = b, where L is operator over V.
        '''
        # Cook up action based on s
        if s > 0:
            def action(self, b):
                x, nsolves, niters = self.apply_negative_power(b, s)
                self.nsolves += nsolves
                self.niters += niters

                return x
        else:
            # This is based on L^s = L^{s+1-1} = L^{(1+s)/2}*L^{-1}*L^{(1+s)/2}
            def action(self, b):
                # L
                A = self.shifted_operator(0.)
                # The first fraction apply
                x0, nsolves, niters = self.apply_negative_power(b, 0.5*(1 + s))
                self.nsolves =+ nsolves
                self.niters += niters
                
                # L action
                b1 = A*x0
                # The second fraction apply
                x, nsolves, niters = self.apply_negative_power(b1, 0.5*(1 + s))
                self.nsolves =+ nsolves
                self.niters += niters
            
                return x
        self.action = action
        
        # This are really only for computing k
        self.V = V
        self.mesh_hmin = MPI.min(V.mesh().mpi_comm(), V.mesh().hmin())
        # Monitors
        self.nsolves, self.niters = 0, 0

    def __call__(self, b):
        '''Apply to b'''
        return self.action(self, b)
        
    def apply_negative_power(self, b, beta):
        '''Bonito and Pesciak's paper on fractional powers of elliptic operator'''
        assert between(beta, (0, 1))
        
        x = self.create_vec(1); x.zero()
        xk = x.copy()
        # Adapt
        k = self.compute_k(beta, b.size(), self.mesh_hmin)

        M = int(ceil(pi**2/(4.*beta*k**2)))
        N = int(ceil(pi**2/(4*(1 - beta)*k**2)))

        nsolves = 0
        iter_count = 0
        for l in range(-M, N+1):
            nsolves += 1
            
            yl = l*k
            shift = exp(-2.*yl)

            A = self.shifted_operator(shift)
            # Keep track of number of iteration in inner solves
            count, xk = self.solve_shifted_problem(A, xk, b)
            iter_count += count
            
            x.axpy(exp(2*yl*(beta - 1.)), xk)
        x *= 2*k*sin(pi*beta)/pi

        return x, nsolves, iter_count

    # API that the child has to implement ---------------------------
    def compute_k(self, fractionality, space_dim, mesh_size):
        raise NotImplementedError()

    def shifted_operator(self, shift):
        raise NotImplementedError()

    def solve_shifted_problem(self, operator, x, vector):
        raise NotImplementedError()

    # Implementation of cbc.block API --------------------------------

    def matvec(self, b):
        return self(b)

    @vec_pool
    def create_vec(self, dim=1):
        return Function(self.V).vector()

######################################################################
# Specializations
######################################################################

def Zero(V):
    '''Zero element of V'''
    return Constant(zero(V.ufl_element().value_shape()))

class BP_HsNorm_Base(BP_Operator_Base):
    '''Hs norms using direct solver'''
    @classmethod
    def prepare_bcs(cls, V, bcs):
        '''whatever -> DirichletBC'''
        try:
            return [prepare_bcs(cls, V, b) for b in bcs]
        except TypeError:
            pass
        
        if bcs is None or isinstance(bcs, DirichletBC):
            return bcs

        return DirichletBC(V, Zero(V), bcs)
    
    def __init__(self, V, bcs, s, parameters):
        '''Children need to give A, I, ... '''
        self.bcs = BP_HsNorm_Base.prepare_bcs(V, bcs)
        # A dummy right hand side for symmetric assembly
        self.L = inner(Zero(V), TestFunction(V))*dx
        # Get stuff for specialization
        BP_Operator_Base.__init__(self, s, V)
        
        # Cook up API functions
        parameters = parameters.copy()
        # quadrature
        f = parameters.pop('k')
        if isinstance(f, (int, float)):
            compute_k = (lambda s, N, h: f)
        else:
            assert isinstance(f, FunctionType)
            assert f.func_code.co_argcount == 3
            compute_k = f 
        self.compute_k = compute_k

        # Solver, just for fun trying to avoid assignments
        # Direct + matrix
        solver = parameters.pop('solver')
        if solver == 'lu':
            # LU
            solve_shifted_problem = lambda A, x, b: (solve(A, x, b), x)
            # System
            shifted_operator = \
                lambda self, shift: assemble_system(self.A + Constant(shift)*self.I,
                                                    self.L,
                                                    self.bcs)[0]
        # Cholesky from cbc.block
        elif solver == 'cholesky':
            # Solver
            solve_shifted_problem = lambda A, x, b: (1, (lambda: Cholesky(A))()*b)
            # System
            shifted_operator = \
                               lambda self, shift: assemble_system(self.A + Constant(shift)*self.I,
                                                                   self.L,
                                                                   self.bcs)[0]
        # Iterative
        else:
            assert solver == 'iterative'
            self.S = None
            self.solver = None
            
            # Action of the shifted operator
            def shifted_operator(self, shift):
                # Update and return
                if self.S is not None:
                    self.S.shift = shift
                    return self.S

                # Setup
                A, _ = assemble_system(self.A, self.L, self.bcs)
                I, _ = assemble_system(self.I, self.L, self.bcs)

                class ShiftedOperator(LinearOperator):
                    def __init__(self, shift):
                        self.shift = shift
                        LinearOperator.__init__(self, Function(V).vector(), Function(V).vector())
                        
                    def mult(self, x, y):
                        z = I*x
                        A.mult(x, y)
                        y.axpy(self.shift, z)
                # One instance of it
                S = ShiftedOperator(shift)
                self.S = S

                # Setup the solver here as well
                # Take A as a preconditioner
                Bmat, _ = assemble_system(self.A + self.I, self.L, self.bcs)
                B = PETScPreconditioner('hypre_amg')
                # Should be SPD
                solver = PETScKrylovSolver('cg', B)
                solver.set_operators(S, Bmat)
                solver.parameters.update(parameters.pop('krylov_parameters'))
                
                self.solver = solver

                # Done
                return S

            def solve_shifted_problem(A, x, b):
                assert self.solver is not None
                niters = self.solver.solve(x, b)
                return niters, x
            
        # Assign bodies
        self.solve_shifted_problem_ = solve_shifted_problem
        self.shifted_operator_ = shifted_operator
        
    # BP_Operator_Base interface
    
    def compute_k(self, s, N, h):
        return self.compute_k(s, N, h)

    def shifted_operator(self, shift):
        '''Return a matrix'''
        return self.shifted_operator_(self, shift) 

    def solve_shifted_problem(self, A, x, b):
        '''Direct by default'''
        return self.solve_shifted_problem_(A, x, b)

######################################################################
# Concrete
######################################################################

class BP_H10Norm(BP_HsNorm_Base):
    '''Dirichlet fractional Laplacian'''
    def __init__(self, V, bcs, s, params):
        assert bcs is not None

        u, v = TrialFunction(V), TestFunction(V)
        h = CellSize(V.mesh())

        if V.ufl_element().family() == 'Discontinuous Lagrange':
            assert V.ufl_element().value_shape() == ()
            assert V.ufl_element().degree() == 0

            h_avg = avg(h)            
            a = h_avg**(-1)*dot(jump(v), jump(u))*dS + h**(-1)*dot(u, v)*ds
        else:
            a = inner(grad(u), grad(v))*dx

        BP_HsNorm_Base.__init__(self, V, bcs, s, params)
        self.A = a
        self.I = inner(u, v)*dx

        
class BP_H1Norm(BP_HsNorm_Base):
    '''Neumannn fractional Helmholtz'''
    def __init__(self, V, s, params):

        u, v = TrialFunction(V), TestFunction(V)
        h = CellSize(V.mesh())

        if V.ufl_element().family() == 'Discontinuous Lagrange':
            assert V.ufl_element().value_shape() == ()
            assert V.ufl_element().degree() == 0

            h_avg = avg(h)
            a = h_avg**(-1)*dot(jump(v), jump(u))*dS + inner(u, v)*dx
        else:
            a = inner(grad(u), grad(v))*dx + inner(u, v)*dx

        bcs = None
        BP_HsNorm_Base.__init__(self, V, bcs, s, params)
        self.A = a
        self.I = inner(u, v)*dx

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitIntervalMesh, FunctionSpace, Expression, assemble
    from dolfin import plot, interactive, Function, interpolate, errornorm, ln
    from dolfin import UnitSquareMesh
    
    s = 0.5
    k = 1.
    
    params = {'k': 0.4,
              'solver': 'iterative',
              'krylov_parameters': {'relative_tolerance': 1E-8,
                                    'absolute_tolerance': 1E-8,
                                    'convergence_norm_type': 'true',
                                    'monitor_convergence': False}}
    if True:
        # f = Expression('sin(k*pi*x[0])', k=k, degree=4)
        # u_exact = Expression('sin(k*pi*x[0])/pow(pow(k*pi, 2), s)', s=s, k=k, degree=4)

        f = Expression('sin(k*pi*x[0])*sin(l*pi*x[1])', k=k, l=2*k, degree=4)
        u_exact = Expression('sin(k*pi*x[0])*sin(l*pi*x[1])/pow(pow(k*pi, 2) + pow(l*pi, 2), s)',
                             s=s, k=k, l=2*k, degree=4)

        
        # f = Expression('sin(k*pi*x[0])', k=k, degree=4)
        # u_exact = Expression('sin(k*pi*x[0])/pow(pow(k*pi, 2), s)', s=s, k=k, degree=4)
        
        
        get_bcs = lambda V: DirichletBC(V, Constant(0), 'on_boundary')
        get_B = lambda V, bcs, s: BP_H10Norm(V, bcs, s, params=params)
    else:
        f = Expression('cos(k*pi*x[0])', k=k, degree=4)
        u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)
        get_bcs = lambda V: None
        get_B = lambda V, bcs, s: BP_H1Norm(V, s, params=params)
    
    e0 = None
    h0 = None
    for n in [2**i for i in range(2, 10)]: #[2**i for i in range(5, 13)]:
        # mesh = UnitIntervalMesh(n)
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, 'CG', 1)
        bcs = get_bcs(V)

        B = get_B(V, bcs, s)

        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        b = assemble(inner(f, v)*dx)

        if bcs is not None: bcs.apply(b)

        x = B*b

        df = Function(V, x)
        h = mesh.hmin()
        e = errornorm(u_exact, df, 'L2', degree_rise=3)
        if e0 is None:
            rate = np.nan
        else:
            rate = ln(e/e0)/ln(h/h0)

        e0 = e
        h0 = float(h)
        nsolves, niters_per_solve = B.nsolves, float(B.niters)/B.nsolves 
        
        print '%d | %.2E %.4E %.2f [%d(%.4f)]' % (V.dim(), mesh.hmin(), e0, rate, nsolves, niters_per_solve)

    u_exact = interpolate(u_exact, V)
    u_exact.vector().axpy(-1, df.vector())

    plot(u_exact)
    interactive()

    # xx = V.tabulate_dof_coordinates()
    
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(xx, u_exact.vector().array(), label='exact')
    # plt.plot(xx, df.vector().array(), label='numeric')
    # plt.legend(loc='best')
    # plt.show()

    
