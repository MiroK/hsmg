from dolfin import FunctionSpace, interpolate, Expression, PETScMatrix
from dolfin import UnitSquareMesh, FiniteElement, UnitIntervalMesh, near
from dolfin import DomainBoundary, CompiledSubDomain

from hsmg.restriction import restriction_matrix, Dirichlet_dofs
from hsmg.hierarchy import by_refining
from hsmg.utils import to_csr_matrix
import numpy as np


def check(seed, elm, f, nlevels=6):
    '''Restriction should work on polynomials of the fem space degree'''

    hierarchy = by_refining(seed, nlevels)
    V = FunctionSpace(hierarchy[0], elm)

    # As petsc
    Rs = restriction_matrix(V, hierarchy, PETScMatrix)
    for i in range(len(hierarchy)-1):
        R = Rs[i]
        Vcoarse = FunctionSpace(hierarchy[i+1], elm)
        Vfine = FunctionSpace(hierarchy[i], elm)

        assert (R.size(0), R.size(1)) == (Vcoarse.dim(), Vfine.dim())

        x = interpolate(f, Vfine).vector()
        # What it should be
        Rf = interpolate(f, Vcoarse).vector()

        y = Rf.copy();
        R.mult(x, y)

        Rf.axpy(-1, y)

        assert Rf.norm('linf') < 1E-14

    # Scipy
    Rs = map(to_csr_matrix, Rs)
    for i in range(len(hierarchy)-1):
        R = Rs[i]
        Vcoarse = FunctionSpace(hierarchy[i+1], elm)
        Vfine = FunctionSpace(hierarchy[i], elm)

        assert R.shape == (Vcoarse.dim(), Vfine.dim())

        x = interpolate(f, Vfine).vector().array()
        # What it should be
        Rf = interpolate(f, Vcoarse).vector().array()

        y = R.dot(x)
    
        assert np.linalg.norm(Rf-y, np.inf) <  1E-14
    return True


def test_1d_P1():
    mesh = UnitIntervalMesh(2)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    f = Expression('2*x[0]', degree=1)

    assert check(mesh, elm, f, 6)

    
def test_1d_P2():
    mesh = UnitIntervalMesh(2)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 2)
    f = Expression('2*x[0]+x[0]*x[0]', degree=2)

    assert check(mesh, elm, f, 6)

    
def test_2d_P1():
    mesh = UnitIntervalMesh(10)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    f = Expression('x[0]+x[1]', degree=1)

    assert check(mesh, elm, f, 6)

    
def test_2d_P2():
    mesh = UnitIntervalMesh(10)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 2)
    f = Expression('x[0]*x[0]+2*x[1]-x[1]*x[1]', degree=2)

    assert check(mesh, elm, f, 6)


def test_DirichletDofs():
    mesh = UnitIntervalMesh(10)
    hierarchy = by_refining(mesh, 4)
    V = FunctionSpace(mesh, 'CG', 1)
    bdry = DomainBoundary()

    bdry_dofs = Dirichlet_dofs(V, bdry, hierarchy)
    assert all(len(dofs_level) == 2 for dofs_level in bdry_dofs)

    bdry = CompiledSubDomain('near(x[0], 2)')
    bdry_dofs = Dirichlet_dofs(V, bdry, hierarchy)
    assert all(len(dofs_level) == 0 for dofs_level in bdry_dofs)
