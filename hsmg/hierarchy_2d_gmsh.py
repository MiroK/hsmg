from collections import namedtuple

Manifold = namedtuple('manifold', ['nodes', 'boundary'])


def smooth_manifolds(mesh):
    '''
    Break the mesh into manifold by smoothness understood as being connected 
    to two facets.
    '''
    tdim = mesh.topology().dim()

    mesh.init(tdim-1)
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)

    # Mappings for the general algorithm
    nodes = range(mesh.num_entities(tdim))
    node2edges = lambda n, n2e=mesh.topology()(tdim, tdim-1): set(n2e(n))
    edge2nodes = lambda e, e2n=mesh.topology()(tdim-1, tdim): set(e2n(e))
    
    edge_is_smooth = lambda e, e2n=edge2nodes: len(e2n(e)) == 2

    return find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)


def find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth):
    '''
    Let there be a graph with nodes connected by edges. A smooth manifold 
    is a collection of nodes connected together by smooth edges.

    INPUT
    nodes = iterable(int)
    node2edges = int -> set([int])
    edge2nodes = int -> set([int])
    edge_is_smooth = int -> bool

    OUTPUT:
    list of (set(nodes), set(edges) that are boundary)
    '''
    starts, terminals = set(), set()
    # Let's find the possible starts - the idea being that we want to build
    # from the non-smoothe edges
    for node in nodes:
        for edge in node2edges(node):
            if not edge_is_smooth(edge):
                starts.add(node)
                terminals.add(edge)

    # We might have a single loop, then the start does not matter
    if not starts:
        # First cell and one it's facet
        starts, terminals = set((0, )), set()

    manifolds = []
    while starts:
        node = starts.pop()

        manifold = manifold_from(node, node2edges, edge2nodes, terminals)
        # Burn bridges every node is part of only one manifold
        starts.difference_update(manifold.nodes)

        manifolds.append(manifold)
    return manifolds


def manifold_from(node, node2edges, edge2nodes, terminals):
    '''
    Grow the manifold from node. Manifold is bounded by a subset of terminal 
    edges

    INPUT
    node = int
    node2edges = int -> set([int])
    edge2nodes = int -> set([int])
    edge_is_smooth = set([int])
    
    OUTPUT:
    (set([int]), set([int])) given as named tuple nodes/boundary
    '''
    # We connect nodes with others over non-terminal edges
    next_edges = node2edges(node)
    manifold_bdry = next_edges & terminals
    next_edges.difference_update(manifold_bdry)

    manifold = set((node, ))
    while next_edges:  # No terminals
        next_e = next_edges.pop()
        
        # Nodes connected to it which are new
        connected_nodes = edge2nodes(next_e) - manifold
        if not connected_nodes:
            continue
        # At most 1
        node,  = connected_nodes
        manifold.add(node)
        
        # The connected node may contribute new edges
        new_edges = node2edges(node) - set((next_e, ))  # Don't go back
        # We're really interested only in those that can be links
        manifold_bdry_ = new_edges & terminals
        new_edges.difference_update(manifold_bdry_)
        
        manifold_bdry.update(manifold_bdry_)
        next_edges.update(new_edges)

    return Manifold(manifold, manifold_bdry)


# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary, File, cells)
    from xii import EmbeddedMesh
    from itertools import chain

    mesh = UnitCubeMesh(2, 2, 2)


    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    # DomainBoundary().mark(f, 1)
    # CompiledSubDomain('near(x[0], 0)').mark(f, 0)
    # CompiledSubDomain('near(x[0], 1)').mark(f, 0)

    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0.0)').mark(f, 1)
    # CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    # CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)

    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0)').mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    #CompiledSubDomain('near(x[0], 1.)').mark(f, 1)


    mesh = EmbeddedMesh(f, 1)

    print len(smooth_manifolds(mesh))
