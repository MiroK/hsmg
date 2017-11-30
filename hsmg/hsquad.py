from dolfin import SubDomain, CompiledSubDomain, between, Constant
from dolfin import DirichletBC, inner, grad, dx, assemble_system
from dolfin import TrialFunction, TestFunction
from dolfin import CellSize, avg, dot, jump, dS, ds
from dolfin import Vector, sqrt, MPI, solve

from block.object_pool import vec_pool
from block.block_base import block_base

from math import sin, pi, sqrt, exp, ceil
import numpy as np


class BPOperatorBase(block_base):
    '''
    TODO
    '''
    def __init__(self, L, bcs, s, params):
        # Same function space for a 
        V = set(arg.function_space() for arg in L.arguments())
        assert len(V) == 1
        V = V.pop()
        # Limit to scalar valued functions for now
        assert V.dolfin_element().value_rank() == 0

        bcs = BPOperatorBase.prepare_boundary_conditions(bcs, V)

        # Keep s to where we know it works
        assert between(s, (-1, 1.))
        
        u, v = TrialFunction(V), TestFunction(V)
        b = inner(Constant(0), v)*dx
        
        # The shifted operator
        self.identity = inner(u, v)*dx
        self.operator = L
        self.shifted_operator = lambda shift: self.operator + Constant(exp(-2.*shift))*self.identity

        self.dummy_rhs = b     # For use with assemble system
        self.bcs = bcs         # To apply when shift is assembled?
        self.s = s             # Fractionality
        # s together with size and mesh size might be things related to
        # how number of quadrature nodes is computed
        self.size = V.dim()
        self.mesh_size = MPI.min(V.mesh().mpi_comm(), V.mesh().hmin())
        self.params = params

    @classmethod
    def prepare_boundary_conditions(cls, bcs, V):
        # Do nothings
        if bcs is None:
            return bcs

        if isinstance(bcs, DirichletBC):
            return bcs
        #
        if isinstance(bcs, (CompiledSubDomain, SubDomain)):
            return DirichletBC(V, Constant(0), bcs)

        # Recurse
        return [prepare_boundary_conditions(cls, bc, V) for bc in bcs]

    def apply(self, b, beta):
        assert between(beta, (0, 1))
        
        x = self.create_vec(1)
        x.zero()
        xk = x.copy()

        k = self.params['k'](self.size, self.mesh_size)

        M = int(ceil(pi**2/(4.*beta*k**2)))
        N = int(ceil(pi**2/(4*(1 - beta)*k**2)))

        count = 0
        e0 = None
        for l in range(-M, N+1):
            count += 1
            yl = l*k
            # print self.operator(yl)
            A, _ = assemble_system(self.shifted_operator(yl),
                                   self.dummy_rhs,
                                   self.bcs)
            #print np.linalg.norm(A.array())
            solve(A, xk, b)
            #print 'factor', exp(2*yl*beta),
            #print 'shift', exp(2*yl)
            
            x.axpy(exp(2*yl*(beta - 1.)), xk)
        
        x *= 2*k*sin(pi*beta)/pi
        
        return x, count

    # Implementation of cbc.block API --------------------------------
    
    def matvec(self, b):
        if self.s > 0:
            x, count = self.apply(b, self.s)
            self.nsolves = count
            return x
        else:
            A, _ = assemble_system(self.operator, self.dummy_rhs, self.bcs)

            z, count0 = self.apply(b, 0.5*(1+self.s))
            
            y = A*z
            
            x, count1 = self.apply(y, 0.5*(1+self.s))

            self.nsolves = count0 + count1
            
            return x

    @vec_pool
    def create_vec(self, dim=1):
        return Vector(None, self.size)
    
# Specializations


class BP_DirichletLaplacian(BPOperatorBase):
    '''
    TODO
    '''
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
        
        BPOperatorBase.__init__(self, a, bcs, s, params)

        
class BP_NeumannHelmholtz(BPOperatorBase):
    '''
    TODO
    '''
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
        BPOperatorBase.__init__(self, a, bcs, s, params)

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitIntervalMesh, FunctionSpace, Expression, assemble
    from dolfin import plot, interactive, Function, interpolate, errornorm, ln

    s = 0.5

    params={'k': lambda size, h: 1/4.}
    if False:
        k = 1.
        f = Expression('sin(k*pi*x[0])', k=k, degree=4)
        u_exact = Expression('sin(k*pi*x[0])/pow(pow(k*pi, 2), s)', s=s, k=k, degree=4)
        get_bcs = lambda V: DirichletBC(V, Constant(0), 'on_boundary')
        get_B = lambda V, bcs, s: BP_DirichletLaplacian(V, bcs, s, params=params)
    else:
        k = 1.
        f = Expression('cos(k*pi*x[0])', k=k, degree=4)
        u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)
        get_bcs = lambda V: None
        get_B = lambda V, bcs, s: BP_NeumannHelmholtz(V, s, params=params)
    
    e0 = None
    h0 = None
    for n in [2**i for i in range(5, 13)]:
        mesh = UnitIntervalMesh(n)
        V = FunctionSpace(mesh, 'DG', 0)
        bcs = get_bcs(V)

        B = get_B(V, bcs, s)

        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        b = assemble(inner(f, v)*dx)

        if bcs is not None:
            bcs.apply(b)

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

        print '%.2E %.4E %.2f [%d]' % (mesh.hmin(), e0, rate, B.nsolves)

    u_exact = interpolate(u_exact, V)
    #u_exact.vector().axpy(-1, df.vector())

    xx = V.tabulate_dof_coordinates()
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xx, u_exact.vector().array(), label='exact')
    plt.plot(xx, df.vector().array(), label='numeric')
    plt.legend(loc='best')
    plt.show()
