from hsmg.hierarchy import by_refining
from hsmg.macro_element import macro_dofmap, vertex_patch

from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from dolfin import EdgeFunction, CompiledSubDomain, BoundaryMesh
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import FunctionSpace, FiniteElement, DomainBoundary
from dolfin import triangle, interval, tetrahedron

import numpy as np
import random
import pytest


def test_hierarchy_dm():
    mesh = UnitIntervalMesh(5)
    hierarchy = by_refining(mesh, 4)
    V = FunctionSpace(hierarchy[0], 'CG', 1)
    ds = macro_dofmap(1, V, hierarchy)

    for mesh, d in zip(hierarchy, ds):
        assert FunctionSpace(mesh, 'CG', 1).dim() == len(d)

        
# --------------------------------------------------------------------


def interval3d():
    '''Mesh for testing line in 3d embedding'''
    mesh = UnitCubeMesh(10, 10, 10)
    f = EdgeFunction('size_t', mesh, 0)
    CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(f, 1)
    return EmbeddedMesh(mesh, f, 1).mesh


def interval2d():
    '''Mesh for testing line in 2d embedding'''
    return BoundaryMesh(UnitSquareMesh(10, 10), 'exterior')


def triangle3d():
    '''Mesh for testing triangle in 3d embedding'''
    return BoundaryMesh(UnitCubeMesh(10, 10, 10), 'exterior')

        
@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron,
                                  interval2d, interval3d, triangle3d])
@pytest.mark.parametrize('degree', [0, 1, 2])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_DG(cell, degree, level):
    '''DG takes all its pathch dofs'''
    meshes = {interval: UnitIntervalMesh(100),
              triangle: UnitSquareMesh(10, 10),
              tetrahedron: UnitCubeMesh(4, 4, 4)}

    mesh = meshes[cell] if cell in meshes else cell()

    elm = FiniteElement('Discontinuous Lagrange', mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, elm)

    i = random.randint(0, mesh.num_vertices()-1)
    # Extract macro element from map
    dofs = set(macro_dofmap(level, V, mesh)[i])
    patch = vertex_patch(mesh, i, level)
    # Count checks out
    assert len(dofs) == len(patch)*V.element().space_dimension()
    # They are the right ones
    dm = V.dofmap()
    assert dofs == set(sum((dm.cell_dofs(c).tolist() for c in patch), []))

    
@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_CG1(cell, level):
    '''CG1 on level is all CG1 on previous level'''
    meshes = {interval: UnitIntervalMesh(100),
              triangle: UnitSquareMesh(10, 10),
              tetrahedron: UnitCubeMesh(8, 8, 8)}

    mesh = meshes[cell] if cell in meshes else cell()
    
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, elm)

    # To avoid corner cases I only look at center point
    X = mesh.coordinates()
    center = np.array([0.5]*mesh.geometry().dim())
    i = min(range(len(X)), key = lambda i: np.linalg.norm(X[i] - center))
        
    dofs = set(macro_dofmap(level, V, mesh)[i])
    
    if level == 1:
        assert len(dofs) == 1, dofs
        return
        
    patch = vertex_patch(mesh, i, level-1)
    dm = V.dofmap()
    dofs_ = set(sum((dm.cell_dofs(c).tolist() for c in patch), []))

    assert dofs == dofs_, (dofs, dofs_)


@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_CG2(cell, level):
    meshes = {interval: UnitIntervalMesh(100),
              triangle: UnitSquareMesh(10, 10),
              tetrahedron: UnitCubeMesh(8, 8, 8)}

    mesh = meshes[cell] if cell in meshes else cell()
    
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, elm)

    gdim = mesh.geometry().dim()
        
    # To avoid corner cases I only look at center point
    X = mesh.coordinates()
    center = np.array([0.5]*gdim)
    i = min(range(len(X)), key = lambda i: np.linalg.norm(X[i] - center))
            
    dofs = set(macro_dofmap(level, V, mesh)[i])

    mesh.init(gdim - 1, gdim)
    mesh.init(gdim, gdim - 1)
    c2f = mesh.topology()(gdim, gdim-1)
    f2c = mesh.topology()(gdim-1, gdim)
        
    # Remove dofs of facets on bouding surface
    patch = vertex_patch(mesh, i, level)
    dm = V.dofmap()

    dofs_ = set(sum((dm.cell_dofs(c).tolist() for c in patch), []))
    bdry = set()
    for cell in patch:
        cell_dofs = dm.cell_dofs(int(cell))
        # Is bounding facet iff it is not shared by cells of patch
        for local, facet in enumerate(c2f(int(cell))):
            other_cell = (set(f2c(facet)) - set([cell])).pop()
            if other_cell not in patch:
                outside = dm.tabulate_facet_dofs(local)
                [bdry.add(cell_dofs[i]) for i in outside]

    dofs_ = dofs_ - bdry

    assert dofs == dofs_, (dofs-dofs_, dofs_-dofs)
