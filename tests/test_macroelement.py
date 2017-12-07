from hsmg.hierarchy import by_refining
from hsmg.macro_element import macro_dofmap, vertex_patch
from hsmg.restriction import Dirichlet_dofs

from dolfin import (EdgeFunction, CompiledSubDomain, BoundaryMesh,
                    UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
                    FunctionSpace, FiniteElement, DomainBoundary,
                    triangle, interval, tetrahedron,
                    RectangleMesh, BoxMesh, Point)

import numpy as np
import random
import pytest


def test_hierarchy_dm():
    '''Sanity chacks, we get a patch for every vertex'''
    mesh = UnitIntervalMesh(5)
    hierarchy = by_refining(mesh, 4)
    V = FunctionSpace(hierarchy[0], 'CG', 1)
    ds = macro_dofmap(1, V, hierarchy)

    for mesh, d in zip(hierarchy, ds):
        # Because there are no bcs
        assert FunctionSpace(mesh, 'CG', 1).dim() == len(d)

        
def test_hierarchy_dm_bcs():
    '''Removing bcs'''
    mesh = UnitSquareMesh(8, 8)
    hierarchy = by_refining(mesh, 4)
    bdry = DomainBoundary()
    
    V = FunctionSpace(hierarchy[0], 'CG', 2)
    
    bdry_dofs = Dirichlet_dofs(V, bdry, hierarchy)
    ds = macro_dofmap(1, V, hierarchy, bdry_dofs)

    for mesh, d, bdofs in zip(hierarchy, ds, bdry_dofs):
        # No boundary dof is present in the macro elemnt of any vertex
        assert not any(set(macro_el) & set(bdofs) for macro_el in d)
        # Note that for P1 there will be less patches then maps because
        # bdry vertex macro_element with bcs is []
        
# --------------------------------------------------------------------

def interval2d():
    '''Mesh for testing line in 2d embedding'''
    return BoundaryMesh(RectangleMesh(Point(-1, 0.5), Point(1, 1), 32, 32),
                        'exterior')


def triangle3d():
    '''Mesh for testing triangle in 3d embedding'''
    return BoundaryMesh(BoxMesh(Point(-1, -1, 0.5), Point(1, 1, 1), 16, 16, 16),
                        'exterior')

        
@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron,
                                  interval2d, triangle3d])
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

    
@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron,
                                  interval2d, triangle3d])
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


@pytest.mark.parametrize('cell', [interval, triangle, tetrahedron,
                                  interval2d, triangle3d])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_CG2(cell, level):
    meshes = {interval: UnitIntervalMesh(100),
              triangle: UnitSquareMesh(10, 10),
              tetrahedron: UnitCubeMesh(8, 8, 8)}

    mesh = meshes[cell] if cell in meshes else cell()
    
    elm = FiniteElement('Lagrange', mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, elm)

    gdim = mesh.geometry().dim()        
    # To avoid corner cases I only look at center p
    X = mesh.coordinates()
    center = np.array([0.5]*gdim)
    i = min(range(len(X)), key = lambda i: np.linalg.norm(X[i] - center))
            
    dofs = set(macro_dofmap(level, V, mesh)[i])

    tdim = mesh.topology().dim()
    
    mesh.init(tdim - 1, tdim)
    mesh.init(tdim, tdim - 1)
    c2f = mesh.topology()(tdim, tdim-1)
    f2c = mesh.topology()(tdim-1, tdim)
        
    # # Remove dofs of facets on bouding surface
    patch = vertex_patch(mesh, i, level)
    dm = V.dofmap()

    print patch
    dofs_ = set(sum((dm.cell_dofs(c).tolist() for c in patch), []))
    bdry = set()
    for cell in patch:
        cell_dofs = dm.cell_dofs(int(cell))
        # Is bounding facet iff it is not shared by cells of patch
        for local, facet in enumerate(c2f(cell)):
            other_cell = (set(f2c(facet)) - set([cell])).pop()
            if other_cell not in patch:
                outside = dm.tabulate_facet_dofs(local)
                [bdry.add(cell_dofs[i]) for i in outside]

    dofs_ = dofs_ - bdry

    assert dofs == dofs_, (dofs-dofs_, dofs_-dofs)
