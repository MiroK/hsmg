from dolfin import *
from hsmg.hsquad import InvFHelmholtz
import numpy as np

# This example shows how to plug in solver to fractional Laplacian
set_log_level(WARNING)

def my_solve(A, x, b):
    '''Direct Cholesky'''
    from block.algebraic.petsc import Cholesky, AMG
    from block.iterative import ConjGrad
    
    # Ainv = Cholesky(A)
    Ainv = ConjGrad(A, precond=AMG(A), tolerance=1E-10)
    x = Ainv*b

    return 1, x


k = 2
f = Expression('cos(k*pi*x[0])', k=k, degree=4)

# Solve (-Delta + I)^s u_exact = f
for s in (0.25, 0.5, 0.75, -0.25):
    hs, errors = [], []
    u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)
    
    for n in [8, 16, 32, 64, 128]:
        mesh = UnitIntervalMesh(n)
        
        V = FunctionSpace(mesh, 'CG', 1)

        b = assemble(inner(TestFunction(V), f)*dx)
        # Numeric
        H = InvFHelmholtz(V, s, bcs=None, compute_k=0.5, solve_shifted_problem=my_solve)
        # NOTE: so the map is nodal to dual
        uh = Function(V, H*b)
        
        error = errornorm(u_exact, uh, 'L2')
        hs.append(mesh.hmin())
        errors.append(error)
    hs, errors = map(np.array, (hs, errors))

    rates = np.r_[np.nan, np.log(errors[1:]/errors[:-1])/np.log(hs[1:]/hs[:-1])]

    print '======= s is', s, '======='
    for c in zip(hs, errors, rates):
        print '%.2E %.2E %.2f' % c
