from dolfin import Mesh, FunctionSpace, Cell
from itertools import ifilter
import numpy as np
import operator


def macro_dofmap(size, space, mesh, bdry_dofs=None):
    '''
    For each VERTEX create a set of degrees of freedom which are located
    on macro element of size around the VERTEX. 
    '''
    assert size >= 1
    # Recurse on hierarchy
    if not isinstance(mesh, Mesh):
        assert bdry_dofs is None or len(mesh) == len(bdry_dofs)
        
        hierarchy = mesh
        bdry_dofs_hierarchy = bdry_dofs if bdry_dofs is not None else [None]*len(hierarchy)
        return [macro_dofmap(size,
                             FunctionSpace(mesh, space.ufl_element()),
                             mesh,
                             bdry_dofs)
                for mesh, bdry_dofs in zip(hierarchy, bdry_dofs_hierarchy)]

    # Base case of single mesh, possible with its bdry_dofs
    if bdry_dofs is None:
        return [np.fromiter(macro_element(space, vertex, size), dtype=int)
                for vertex in range(mesh.num_vertices())]
    # With bcs it can happen that the macroelement after removing bcs
    # dofs is empty, in that case it does not enter the map
    else:
        bdry_dofs = set(bdry_dofs)
        maybe = (macro_element(space, vertex, size) - bdry_dofs
                 for vertex in range(mesh.num_vertices()))
        # Remove bdry dofs from macro element, only not empy remaing
        definitely = ifilter(bool, maybe)
        return [np.fromiter(elm, dtype=int) for elm in definitely]


def macro_element(V, vertex, level):
    '''
    Basis functions with supports on a vertex_patch(mesh, vertex, level)
    '''
    assert level >= 1
    patch = vertex_patch(V.mesh(), vertex, level)
    # Get the neighbors of the patch
    rim = vertex_patch(V.mesh(), vertex, level+1) - patch

    # All dofs of the patch - those of the rim are what we want
    dm = V.dofmap()
    patch_dofs = sum((dm.cell_dofs(c).tolist() for c in patch), [])
    rim_dofs = sum((dm.cell_dofs(c).tolist() for c in rim), [])

    return set(patch_dofs) - set(rim_dofs)


def vertex_patch(mesh, vertex, level):
    '''
    A patch of level 1 is formed by cells that share the vertex. Higher 
    level patch is a union of level-1 patches around vertices of level 1.
    '''
    assert level >= 1
    
    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    v2c = mesh.topology()(0, tdim)
    mesh.init(tdim, 0)
    c2v = mesh.topology()(tdim, 0)
    
    patch = set(v2c(vertex))

    all_nodes = reduce(operator.or_, (set(c2v(c)) for c in patch))
    seeds = all_nodes - set([vertex])
    # NOTE, this is of course more elegant with recursion but it becomes
    # slow quite early
    while level > 1:
        # Vertices of the cells in the patch become seed os pathc1
        new_nodes = set()
        while seeds:
            v = seeds.pop()
            new_patch = vertex_patch(mesh, v, 1)
            # The patch grows by the new cells
            for c in new_patch:
                new_nodes.update(set(c2v(c)))
            patch.update(new_patch)
        # Remove from new_nodes the nodes we have seen
        new_nodes.difference_update(all_nodes)
        # These guys are seeds for next round
        seeds.update(new_nodes)
        # And will never be seeds again
        all_nodes.update(seeds)

        level -= 1

    return patch


def cell_patch(mesh, cell):
    '''Union of vertex patches of the cell vertices'''
    tdim = mesh.topology().dim()
    mesh.init(tdim, 0)
    return map(lambda index: Cell(mesh, index),
               reduce(operator.or_,
                      (vertex_patch(mesh, vertex, 1) for vertex in cell.entities(0))))
    
