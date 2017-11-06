from dolfin import FunctionSpace, interpolate, Expression, PETScMatrix
from dolfin import UnitSquareMesh, FiniteElement, UnitIntervalMesh, near
from dolfin import DomainBoundary, CompiledSubDomain, plot
from dolfin import FacetFunction, EdgeFunction, UnitCubeMesh

from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from hsmg.restriction import restriction_matrix, Dirichlet_dofs
from hsmg.hierarchy import by_refining
from hsmg.utils import to_csr_matrix

import numpy as np
import pytest

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

        x = interpolate(f, Vcoarse).vector()
        # What it should be
        Pf = interpolate(f, Vfine).vector()
        # Applying the interpolation matrix gives vector ...
        y = Pf.copy();
        y.zero()
        R.transpmult(x, y)
        
        # That should be the same as direct interpolation
        Pf.axpy(-1, y)

        if Pf.norm('linf') > 1E-14:
            plot(Function(Vfine, Pf), interactive=True)
            
            error = np.abs(Pf.array())
            where = np.argsort(error)[::-1]
            count = Vfine.dim()
            for size, x in zip(error[where],
                               Vfine.tabulate_dof_coordinates().reshape((Vfine.dim(), -1))[where]):
                if size < 1E-14: break
                print size, x
                count -= 1
            print error, '%d/%d' % (count, Vfine.dim())


    return True


def test_1d_P1():
    mesh = UnitIntervalMesh(2)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    f = Expression('2*x[0]', degree=1)

    assert check(mesh, elm, f, 6)

    
def test_2d_P1():
    mesh = UnitSquareMesh(4, 4)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    f = Expression('x[0]+x[1]', degree=1)

    assert check(mesh, elm, f, 4)

    
def test_3d_P1():
    mesh = UnitCubeMesh(2, 2, 2)
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    f = Expression('x[0] + 2*x[1]- x[2]', degree=1)

    assert check(mesh, elm, f, 4)

# --------------------------------------------------------------------

@pytest.mark.parametrize('family', ['Lagrange', 'Discontinuous Lagrange'])        
def test_2d1d(family):
    mesh = UnitSquareMesh(4, 4)    
    gamma = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], x[1])').mark(gamma, 1)

    mesh = EmbeddedMesh(mesh, gamma, 1).mesh
    f = Expression('x[0]+2*x[1]', degree=1)
    elm = FiniteElement(family, mesh.ufl_cell(), 1)
    
    assert check(mesh, elm, f, 4)

    
@pytest.mark.parametrize('family', ['Lagrange', 'Discontinuous Lagrange'])        
def test_3d1d(family):
    mesh = UnitCubeMesh(2, 2, 2)
    gamma = EdgeFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(gamma, 1)

    mesh = EmbeddedMesh(mesh, gamma, 1).mesh
    f = Expression('x[0]+2*x[1]-x[2]', degree=1)
    elm = FiniteElement(family, mesh.ufl_cell(), 1)
    
    assert check(mesh, elm, f, 4)

    
@pytest.mark.parametrize('family', ['Lagrange', 'Discontinuous Lagrange'])        
def test_3d2d(family):
    mesh = UnitCubeMesh(2, 2, 2)
    gamma = FacetFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], 0)').mark(gamma, 1)

    mesh = EmbeddedMesh(mesh, gamma, 1).mesh
    f = Expression('x[0]+2*x[1]-x[2]', degree=1)
    elm = FiniteElement(family, mesh.ufl_cell(), 1)
    
    assert check(mesh, elm, f, 4)
    
# --------------------------------------------------------------------
        
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


def test_DirichletDofsDG():
    mesh = UnitIntervalMesh(10)
    hierarchy = by_refining(mesh, 4)
    V = FunctionSpace(mesh, 'DG', 1)
    # Have to be a bit less strict 
    bdry = CompiledSubDomain('near(x[0]*(1-x[0]), 0, 1E-13)')

    bdry_dofs = Dirichlet_dofs(V, bdry, hierarchy)
    assert all(len(dofs_level) == 2 for dofs_level in bdry_dofs)
