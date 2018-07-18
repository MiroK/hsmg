from collections import namedtuple


Manifold = namedtuple('manifold', ['nodes', 'boundary'])


def smooth_manifolds(mesh):
    '''
    Break an embedded mesh into manifold by smoothness understood as 
    being connected to two facets.

    INPUT:
    mesh = Mesh instance

    OUPUT:
    list([Manifold])
    '''
    assert mesh.geometry().dim() > mesh.topology().dim()
    tdim = mesh.topology().dim()

    mesh.init(tdim-1)
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)

    # Mappings for the general algorithm
    nodes = set(range(mesh.num_entities(tdim)))
    node2edges = lambda n, n2e=mesh.topology()(tdim, tdim-1): set(n2e(n))
    edge2nodes = lambda e, e2n=mesh.topology()(tdim-1, tdim): set(e2n(e))
    
    edge_is_smooth = lambda e, e2n=edge2nodes: len(e2n(e)) == 2

    return find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)


def find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth):
    '''
    Let there be a graph with nodes connected by edges. A smooth manifold 
    is a collection of nodes connected together by smooth edges.

    INPUT:
    nodes = set([int])
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
        # Grow the manifold from node (only adding from nodes)
        manifold = manifold_from(node, nodes, node2edges, edge2nodes, terminals)
        # Burn bridges every node is part of only one manifold
        starts.difference_update(manifold.nodes)

        manifolds.append(manifold)
    return manifolds


def manifold_from(node, nodes, node2edges, edge2nodes, terminals):
    '''
    Grow the manifold from node using nodes. Manifold is bounded by a 
    subset of terminal edges

    INPUT:
    node = int
    nodes = set([int])
    node2edges = int -> set([int])
    edge2nodes = int -> set([int])
    edge_is_smooth = set([int])
    
    OUTPUT:
    Manifold
    '''
    # We connect nodes with others over non-terminal edges
    next_edges = node2edges(node)
    manifold_bdry = next_edges & terminals
    next_edges.difference_update(manifold_bdry)

    manifold = set((node, ))
    while next_edges:  # No terminals
        next_e = next_edges.pop()
        
        # Nodes connected to it which are new
        connected_nodes = edge2nodes(next_e) & nodes - manifold
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
