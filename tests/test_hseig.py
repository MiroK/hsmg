from dolfin import *
from hsmg.hseig import Hs0Eig, HsNorm
from hsmg.utils import from_np_array
import numpy as np
import pytest


def test_identity():
    '''Inverse works'''
    mesh = UnitSquareMesh(5, 5)
    
    V = FunctionSpace(mesh, 'CG', 1)
    H = HsNorm(V, s=0.123)

    x = H.create_vec()
    x.set_local(np.random.rand(x.local_size()))
    
    y = (H**-1)*(H*x)
        
    error = (x-y).norm('linf')

    assert error < 1E-13


@pytest.mark.parametrize('kappa', (1, 0.5, 3))
def test_def_1(kappa):
    '''Agreas with Laplace for s=1'''
    kappa = Constant(kappa)
    mesh = UnitSquareMesh(5, 5)

    V = FunctionSpace(mesh, 'CG', 1)
    H = HsNorm(V, s=1.0, kappa=kappa)

    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(kappa*inner(u, v)*dx + kappa*inner(grad(u), grad(v))*dx)

    x = H.create_vec()
    x.set_local(np.random.rand(x.local_size()))

    y1 = H*x
    y2 = A*x

    error = (y1 - y2).norm('linf')

    assert error < 1E-13

    
def test_def_0():
    '''Agrees with identity for s=0'''
    mesh = UnitSquareMesh(5, 5)

    V = FunctionSpace(mesh, 'CG', 1)
    H = HsNorm(V, s=0.0)

    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(inner(u, v)*dx)

    x = H.create_vec()
    x.set_local(np.random.rand(x.local_size()))

    y1 = H*x
    y2 = A*x

    error = (y1 - y2).norm('linf')
    assert error < 1E-13

    
@pytest.mark.parametrize('make_space', (lambda mesh: FunctionSpace(mesh, 'CG', 1),
                                        lambda mesh: FunctionSpace(mesh, 'DG', 0)))
def test_def(make_space):
    '''Approximation of truth using eigenfunction on (0, 1) cos(k*pi*x)'''
    k = 2
    for s in (0.25, 0.5, 0.75):
        hs, errors = [], []
        for n in [8, 16, 32, 64, 128, 256, 512]:
            mesh = UnitIntervalMesh(n)

            V = make_space(mesh)
            # An eigenfunction
            f = interpolate(Expression('cos(k*pi*x[0])', k=k, degree=1), V).vector()
            # Numeric
            H = HsNorm(V, s=s)
            Hs_norm = f.inner(H*f)
            # From def <(-Delta + I)^s u, v> 
            truth = ((k*pi)**2 + 1)**(s) * 0.5  # 0.5 is form L2 norm of f)

            error = abs(truth - Hs_norm)
            hs.append(mesh.hmin())
            errors.append(error)
        # Decrease
        assert all(np.diff(errors) < 0)
        # Order
        deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
        assert deg > 1


@pytest.mark.parametrize('make_space', (lambda mesh: FunctionSpace(mesh, 'CG', 1),
                                        lambda mesh: FunctionSpace(mesh, 'DG', 0)))        
def test_def(make_space):
    '''Approximation of truth'''
    k = 2

    for s in (0.25, 0.5, 0.75):
        hs, errors = [], []
        for n in [8, 16, 32, 64, 128, 256, 512]:
            mesh = UnitIntervalMesh(n)

            V = make_space(mesh)
            # An eigenfunction
            f = interpolate(Expression('sin(k*pi*x[0])', k=k, degree=1), V).vector()
            # Numeric
            facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
            DomainBoundary().mark(facet_f, 1)
            bcs = [(facet_f, 1)]
            H = Hs0Eig(V, s=s, bcs=bcs)
            Hs_norm = f.inner(H*f)
            # From def <(-Delta + I)^s u, v> 
            truth = (k*pi)**(2*s) * 0.5  # 0.5 is form L2 norm of f)

            error = abs(truth - Hs_norm)
            hs.append(mesh.hmin())
            errors.append(error)
        # Decrease
        assert (all(np.diff(errors) < 0))
        # Order
        deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
        assert (deg > 0.925)

        
def test_def_nodal_dual():
    '''Approximation of truth'''
    for s in (0.25, 0.5, 0.75):
        hs, errors = [], []
        for n in [8, 16, 32, 64, 128, 256, 512]:
            mesh = UnitIntervalMesh(n)

            V = FunctionSpace(mesh, 'CG', 1)

            k = 2
            # An eigenfunction
            f = interpolate(Expression('sin(k*pi*x[0])', k=k, degree=1), V)
            # Numeric
            facet_f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
            DomainBoundary().mark(facet_f, 1)
            bcs = [(facet_f, 1)]
            
            H = Hs0Eig(V, bcs=bcs, s=s)
            dHf = H*f.vector() # This is dual
            # Want nodal
            Hf = Function(V)
            u, v = TrialFunction(V), TestFunction(V)
            solve(assemble(inner(u, v)*dx), Hf.vector(), dHf)
            # From def <(-Delta + I)^s u, v> 
            truth = (k*pi)**(2*s)
            f = truth*f

            error = sqrt(abs(assemble(inner(f - Hf, f - Hf)*dx)))
            hs.append(mesh.hmin())
            errors.append(error)
        # Decrease
        assert (all(np.diff(errors) < 0))
        # Order
        deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
        assert (deg > 1)
