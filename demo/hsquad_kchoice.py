# We think about using InvFHelmholtz in a preconditioner in which case
# the problem needs to be solved only approximately. We are looking for
# a heuristic for k

from dolfin import *
from hsmg.hsquad import InvFHelmholtz
from block.algebraic.petsc import AMG, Cholesky
import numpy as np


set_log_level(WARNING)  # Shut up cbc.block setup info


def get_error(k_value, s, n):
    f = Expression('cos(k*pi*x[0])', k=1, degree=4)
    u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=1, degree=4)
    
    mesh = UnitIntervalMesh(n)
        
    V = FunctionSpace(mesh, 'CG', 1)

    b = assemble(inner(TestFunction(V), f)*dx)
    # Numeric
    H = InvFHelmholtz(V, s, bcs=None, compute_k=k_value)
    # NOTE: so the map is nodal to dual
    uh = Function(V, H*b)
    
    error = errornorm(u_exact, uh, 'L2')
    print '\t', k_value, '->', error, 'using', H.nsolves

# k needs to be small enough to see the conergence in h. However, the error
# of the method seems small for big k already (k=1). So for preconditioning
# it may suffice to run with large (fixed?) k \neq k(mesh size), i.e. fixed
# number of integration points.

if False:
    s = 0.75
    for n in [32, 64, 128, 256, 512, 1024]:
        get_error(k_value=0.55, s=s, n=n)

# Let's see
from hsmg.hseig import HsNorm
from block.iterative import ConjGrad

# so we setup the problem with fractional Laplacian using the Bonito&Pesciak
# operator as preconditioner
def solve_fract_helm(n, s, compute_k):
    mesh = UnitIntervalMesh(n)

    V = FunctionSpace(mesh, 'CG', 1)
    # Operator
    A = HsNorm(V, s)
    # Preconditioner
    def solve_shifted_problem(A, x, b):
        x = Cholesky(A)*b  # Direct; one sweep of AMG is AMG(A)*b
        return (1, x)
    
    B = InvFHelmholtz(V, s, bcs=None, compute_k=compute_k,
                      solve_shifted_problem=solve_shifted_problem)
    
    Ainv = ConjGrad(A, precond=B, tolerance=1E-14)

    x = A.create_vec()
    x.set_local(np.random.rand(x.local_size()))
    
    f = Expression('cos(k*pi*x[0])', k=1, degree=4)
    u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=1, degree=4)
    
    b = assemble(inner(TestFunction(V), f)*dx)
    x = Ainv*b

    uh = Function(V, x)
    error = errornorm(u_exact, uh, 'L2')

    return error, len(Ainv.residuals)-1, B.nsolves


for n in [32, 64, 128, 256, 512, 1024]:
    error, cg_iters, BP_solves = solve_fract_helm(n, s=-0.5, compute_k=10.)
    print 'error = %.4E | #CG = %d | #BP = %d' % (error, cg_iters, BP_solves)
