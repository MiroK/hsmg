from dolfin import (Mesh, FunctionSpace, Cell, MeshFunction, DomainBoundary,
                    SubsetIterator)
from collections import defaultdict
from itertools import ifilter, imap
import numpy as np
import operator


def macro_dofmap(size, space, mesh, bdry_dofs=None):
    '''
    For each VERTEX create a set of degrees of freedom which are located
    on macro element of size around the VERTEX. 
    '''
    assert size >= 1
    # FIXME: 1/4/2020 this is hack making hs(A)MG work
    bdry_dofs = None
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
    # Become aware of the domain boundary is
    f = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
    DomainBoundary().mark(f, 1)
    domain_boundary = set(imap(operator.methodcaller('index'), SubsetIterator(f, 1)))

    if bdry_dofs is None:
        return [np.fromiter(macro_element(space, v, size, domain_boundary), dtype=int)
                for v in range(mesh.num_vertices())]

    # With bcs it can happen that the macroelement after removing bcs
    # dofs is empty, in that case it does not enter the map
    else:
        bdry_dofs = set(bdry_dofs)
        maybe = (macro_element(space, vertex, size, domain_boundary) - bdry_dofs
                 for vertex in range(mesh.num_vertices()))
        # Remove bdry dofs from macro element, only not empy remaing
        definitely = ifilter(bool, maybe)
        return [np.fromiter(elm, dtype=int) for elm in definitely]


def macro_element(V, vertex, level, domain_boundary):
    '''
    Basis functions with supports on interior(vertex_patch(mesh, vertex, level))
    '''
    assert level >= 1
    mesh = V.mesh()
    patch = vertex_patch(mesh, vertex, level)

    # Divide the cells into those that are adjacent to the boundary and not
    bdry_facets_map = patch_boundary(patch, vertex, mesh, domain_boundary)

    dm = V.dofmap()
    dofs = set()

    [dofs.update(set(dm.cell_dofs(cell))) for cell in patch]

    ftdim = mesh.topology().dim() - 1
    # Now remove from all dofs of the cells those that are associated
    # with enties \subeq bdry facets
    for cell in bdry_facets_map:
        cell_dofs = dm.cell_dofs(cell)

        for index in bdry_facets_map[cell]:
            bdry_dof_indices = dm.tabulate_entity_closure_dofs(ftdim, index)
            dofs.difference_update(cell_dofs[bdry_dof_indices])

    return dofs


def patch_boundary(patch, vertex, mesh, domain_boundary):
    '''
    Compute a map which is used to ellimita boundary dofs of the macroelement.
    '''
    topology = mesh.topology()
    tdim = topology.dim()
    mesh.init(tdim, tdim - 1)
    c2f = topology(tdim, tdim-1)

    # We want facets which are connected to only to a single cell of the patch
    facet_cell_map, facet_index_map = {}, {}  # Hold cell local facet indices
    # Compute facet cell connectivity for the patch
    for cell in patch:
        facet_index_map[cell] = {}
        for index, facet in enumerate(c2f(cell)):
            # NOTE: facet is visited at most from two cells. Therefore if
            # I am about to add to an nonempty  I can discard the facet
            if facet in facet_cell_map:
                del facet_cell_map[facet]
            else:
                facet_cell_map[facet] = cell
                facet_index_map[cell][facet] = index

    # Bdry versus interior
    # o----o                o---o 
    # |  / |               / \ / \          
    # |/   |              o---x---o        
    # x----o               \ / \ /        
    #                       o---o
    has_domain_boundary = False
    # Facet is connected to / is a vertex <=> bdry
    if tdim > 1:
        mesh.init(tdim - 1, 0)
        f2v = topology(tdim - 1, 0)
        # The different between first and second case is that we want
        # to keep the INTERIOR dofs of the facets so these facets are removed
        for facet in facet_cell_map.keys():
            if vertex in f2v(facet):
                del facet_cell_map[facet]
                has_domain_boundary = True
    else:
        # x----o make sure that we have not caught the patch defining guy
        for facet in facet_cell_map.keys():
            if vertex == facet:
                del facet_cell_map[facet]
                has_domain_boundary = True
    # Facets on the boundary get to keep their interior dofs!
    if has_domain_boundary:
        for facet in facet_cell_map.keys():
            if facet in domain_boundary:
                del facet_cell_map[facet]

    bdry_map = defaultdict(list)
    for facet, cell in facet_cell_map.iteritems():
        bdry_map[cell].append(facet_index_map[cell][facet])

    return bdry_map

                
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
