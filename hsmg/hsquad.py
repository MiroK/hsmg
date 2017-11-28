from dolfin import SubDomain, CompiledSubDomain, between, Constant
from dolfin import DirichletBC, inner, grad, dx, assemble_system
from dolfin import TrialFunction, TestFunction
from dolfin import CellSize, avg, dot, jump, dS, ds
from dolfin import Vector, sqrt, MPI, solve

from block.object_pool import vec_pool
from block.block_base import block_base

import math

import macro_element
import hs_multigrid
import restriction
import hierarchy
import utils

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
        self.shifted_operator = lambda shift: Constant(math.exp(2*shift))*self.operator + self.identity

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
        
        if isinstance(bcs, (CompiledSubDomain, SubDomain)):
            return DirichletBC(V, Constant(0), bcs)

        # Recurse
        return [prepare_boundary_conditions(cls, bc, V) for bc in bcs]

    def matvec(self, b):
        if self.s > 0:
            return self.apply(b, self.s)
        else:
            A, _ = assemble_system(self.operator, self.dummy_rhs, self.bcs)
            M, _ = assemble_system(self.identity, self.dummy_rhs, self.bcs)

            z = self.apply(b, 0.5*(1-self.s))
            
            y = A*z
            
            # print y.norm('l2')
            x = self.apply(y, 0.5*(1-self.s))
            #print x.norm('l2')
            
            return x
        
    # Implementation of cbc.block API --------------------------------
    def apply(self, b, beta):
        print 'beta', beta
        x = self.create_vec(1)
        x.zero()
        xk = x.copy()

        k = self.params['k'](self.size, self.mesh_size)

        M = int(math.ceil(math.pi**2/(4.*beta*k**2)))
        N = int(math.ceil(math.pi**2/(4*(1 - beta)*k**2)))

        count = 0

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
            
            x.axpy(math.exp(2*beta*yl), xk)
        x *= 2*k*math.sin(math.pi*beta)/math.pi
        return x

    @vec_pool
    def create_vec(self, dim=1):
        return Vector(None, self.size)


class BPDirichletLaplacian(BPOperatorBase):
    '''
    TODO
    '''
    def __init__(self, V, bcs, s, params):
        assert bcs is not None

        u, v = TrialFunction(V), TestFunction(V)
        h = CellSize(V.mesh())
        a = inner(grad(u), grad(v))*dx            
        
        BPOperatorBase.__init__(self, a, bcs, s, params)
        
# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitIntervalMesh, FunctionSpace, Expression, assemble
    from dolfin import plot, interactive, Function, interpolate, errornorm, ln

    s = 0.5

    k = 1.
    f = Expression('sin(k*pi*x[0])', k=k, degree=4)
    u_exact = Expression('sin(k*pi*x[0])/pow(k*pi, 2*s)', s=s, k=k, degree=4)

    e0 = None
    h0 = None
    for n in [2**i for i in range(5, 13)]:
        mesh = UnitIntervalMesh(n)
        V = FunctionSpace(mesh, 'CG', 1)
        bcs = DirichletBC(V, Constant(0), 'on_boundary')

        B = BPDirichletLaplacian(V, bcs, s=s, params={'k': lambda size, h: 1/4.})

        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        b = assemble(inner(f, v)*dx)
        #_, b = assemble_system(a, b, bcs)
        bcs.apply(b)
        
        x = B*b

        df = Function(V, x)
        h = mesh.hmin()
        e = errornorm(u_exact, df, 'L2', degree_rise=3)
        if e0 is None:
            rate = -1
        else:
            rate = ln(e/e0)/ln(h/h0)

        e0 = e
        h0 = float(h)

        print '%.2E %.4E %.2f' % (mesh.hmin(), e0, rate)
    u_exact = interpolate(u_exact, V)
    #u_exact.vector().axpy(-1, df.vector())

    xx = V.tabulate_dof_coordinates()
    for dof in bcs.get_boundary_values().keys():
        print u_exact.vector().array()[dof], xx[dof]
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xx, u_exact.vector().array(), label='exact')
    plt.plot(xx, df.vector().array(), label='numeric')
    plt.legend(loc='best')
    plt.show()
