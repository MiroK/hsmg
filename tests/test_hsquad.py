from hsmg.hsquad import BP_H1Norm, BP_H10Norm
from dolfin import *
import pytest


# FIXME: with DG0 works okay for s > 0, s < 0 gets works the smaller
# s is and it seems that all the error is on the boundary
@pytest.mark.parametrize('solver', ['cholesky', 'iterative'])
@pytest.mark.parametrize('element', [FiniteElement('Lagrange', interval, 1)])
@pytest.mark.parametrize('s_value', [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9])
def test_poisson_1d(solver, s_value, element):
    params = {'k': 0.25, # This should be small enough to get order for all
              'krylov_parameters': {'relative_tolerance': 1E-8,
                                    'absolute_tolerance': 1E-8,
                                    'convergence_norm_type': 'true',
                                    'monitor_convergence': False}}
    params['solver'] = solver

    k = 1.
    f = Expression('sin(k*pi*x[0])', k=k, degree=4)
    u_exact = Expression('sin(k*pi*x[0])/pow(pow(k*pi, 2), s)', s=s_value, k=k, degree=4)

    get_bcs = lambda V: DirichletBC(V, Constant(0), 'on_boundary')
    get_B = lambda V, bcs, s: BP_H10Norm(V, bcs, s, params=params)

    e0 = None
    h0 = None
    for n in [2**i for i in range(2, 5)]:
        mesh = UnitIntervalMesh(n)
        
        V = FunctionSpace(mesh, element)
        bcs = get_bcs(V)

        B = get_B(V, bcs, s_value)

        v = TestFunction(V)
        b = assemble(inner(f, v)*dx)
        if bcs is not None: bcs.apply(b)

        x = B*b

        df = Function(V, x)
        h = mesh.hmin()
        e = errornorm(u_exact, df, 'L2', degree_rise=3)
        if e0 is None:
            rate = 0.
        else:
            rate = ln(e/e0)/ln(h/h0)
        print rate
        e0 = e
        h0 = float(h)
    print rate
    assert any(((element.degree() == 1 and rate > 1.9),
                (element.degree() == 0 and rate > 0.9)))


@pytest.mark.parametrize('solver', ['cholesky', 'iterative'])
@pytest.mark.parametrize('element', [FiniteElement('Lagrange', interval, 1),
                                     FiniteElement('Discontinuous Lagrange', interval, 0)])
@pytest.mark.parametrize('s_value', [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9])
def test_helmholtz_1d(solver, s_value, element):
    params = {'k': 0.25, # This should be small enough to get order for all
              'krylov_parameters': {'relative_tolerance': 1E-8,
                                    'absolute_tolerance': 1E-8,
                                    'convergence_norm_type': 'true',
                                    'monitor_convergence': False}}
    params['solver'] = solver

    k = 1.
    f = Expression('cos(k*pi*x[0])', k=k, degree=4)
    u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)',
                         s=s_value, k=k, degree=4)

    get_B = lambda V, bcs, s: BP_H1Norm(V, s, params=params)

    e0 = None
    h0 = None
    for n in [2**i for i in range(2, 5)]:
        mesh = UnitIntervalMesh(n)
        
        V = FunctionSpace(mesh, element)
        bcs = None

        B = get_B(V, bcs, s_value)

        v = TestFunction(V)
        b = assemble(inner(f, v)*dx)

        x = B*b

        df = Function(V, x)
        h = mesh.hmin()
        e = errornorm(u_exact, df, 'L2', degree_rise=3)
        if e0 is None:
            rate = 0
        else:
            rate = ln(e/e0)/ln(h/h0)
        print rate
        e0 = e
        h0 = float(h)
    print rate
    assert any(((element.degree() == 1 and rate > 1.9),
                (element.degree() == 0 and rate > 0.9)))


# FIXME: DG0 does not work for neither positive nor negarive s
@pytest.mark.parametrize('solver', ['cholesky', 'iterative'])
@pytest.mark.parametrize('element', [FiniteElement('Lagrange', triangle, 1)])
@pytest.mark.parametrize('s_value', [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9])
def test_helmholtz_2d(solver, s_value, element):
    params = {'k': 0.25, # This should be small enough to get order for all
              'krylov_parameters': {'relative_tolerance': 1E-8,
                                    'absolute_tolerance': 1E-8,
                                    'convergence_norm_type': 'true',
                                    'monitor_convergence': False}}
    params['solver'] = solver

    k = 1.
    f = Expression('cos(k*pi*x[0])*cos(k*pi*x[1])', k=k, degree=4)
    u_exact = Expression('cos(k*pi*x[0])*cos(k*pi*x[1])/pow(2*pow(k*pi, 2) + 1, s)',
                         s=s_value, k=k, degree=4)


    get_B = lambda V, bcs, s: BP_H1Norm(V, s, params=params)

    e0 = None
    h0 = None
    for n in [2**i for i in range(2, 5)]:
        mesh = UnitSquareMesh(n, n)
        
        V = FunctionSpace(mesh, element)
        bcs = None

        B = get_B(V, bcs, s_value)

        v = TestFunction(V)
        b = assemble(inner(f, v)*dx)

        x = B*b

        df = Function(V, x)
        h = mesh.hmin()
        e = errornorm(u_exact, df, 'L2', degree_rise=3)
        if e0 is None:
            rate = 0
        else:
            rate = ln(e/e0)/ln(h/h0)
        print rate
        e0 = e
        h0 = float(h)
    print rate
    assert element.degree() == 1 and rate > 1.0
