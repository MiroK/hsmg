from collections import deque
from itertools import ifilter, repeat
from dolfin import near
import numpy as np


def coarsen_1d_mesh(mesh, TOL=1E-13):
    '''Coarse a manifold mesh of lines'''
    # The idea is that a work horse is refinement of straight segments
    segments = find_segments(mesh, TOL)
    
    coarse_segments, success = [], False
    for segment in segments:
        # Work horse
        csegment, coarsened = coarsen_segment(segment)
        # Might fail but some other segment could succeed
        if coarsened:
            coarse_segments.append(csegment)
        # so we proceeed with the original
        else:
            coarse_segments.append(segment)
        # At least one success
        success = success or coarsened
    # No?
    if not success:
        return mesh, success
    # Stich together
    return mesh_from_segments(coarse_segments), success


def find_segments(mesh, TOL=1E-13):
    '''
    Produce representation of straight segments of mesh of the form 
    [[v0, ..., vn],  [vn, ...., vm]] with v_i the vertex indices
    '''
    # The idea is that any branch in the mesh must for such consist of
    # at least one straight segment. So we make branches which can be walked
    # and the walk broken once the orientation changes. It is really not
    # necessary to do it this way but it simplifies stiching the mesh
    branches = find_branches(mesh)
    
    return sum((break_to_segments(branch, mesh, TOL) for branch in branches), [])


def break_to_segments(branch, mesh, TOL):
    '''We have a linked list of pairs'''
    x = mesh.coordinates()

    segments = []
    while branch:
        edge = branch.pop()
        v0, v1 = edge
        
        t1 = x[v0] - x[v1]
        t1 /= np.linalg.norm(t1)

        segment = [v0]  
        # Connected to it?
        while branch:
            edge = branch.pop()
            v1_, v2 = edge
            assert v1 == v1_   # The link property
            
            t2 = x[v1] - x[v2]
            t2 /= np.linalg.norm(t2) 

            ip = abs(np.dot(t1, t2))
            if abs(ip - 1) < TOL:
                segment.append(v2)
                v1 = v2        # The link
            else:
                # If there is work then the next segment starts from edge
                break
        segments.append(segment)
    return segments


def find_branches(mesh):
    '''
    A branch is a list of tuples (that are vertex indices) representing 
    edges of the mesh such that the 
    i) if first and last edge are different then both have a vertex which 
    is either connected to one edge or to more than two edges
    ii) a loop
    '''
    # The relevant connectivity
    # Precompute connectivity, tangent vectors
    mesh.init(0, 1)
    e2v = mesh.topology()(1, 0)
    v2e = mesh.topology()(0, 1)

    start_edges, start_vertices = [], []
    # Let's find the start edges
    for v in range(mesh.num_vertices()):
        # Bdry point
        ne = len(v2e(v))
        if ne == 1:
            start_edges.append(v2e(v)[0])
            start_vertices.append(v)
        # Bifurcation
        if ne > 2:
            start_edges.extend(list(v2e(v)))
            start_vertices.extend([v]*ne)
            print '!'
            
    # We might have a single loop, then the start does not matter
    if len(start_edges) == 0:
        # First edge, and one of its vertices
        start_edges, start_vertices = [0], [e2v(0)[0]]

    # Want to represent edge connectivity by pop-able strucure
    e2v = {e: list(e2v(e)) for e in range(mesh.num_cells())}
    # For termination checking
    terminals = set(start_vertices)  # Unique

    print start_edges, start_vertices
    branches = []
    while start_edges:
        vertex, edge = start_vertices.pop(), start_edges.pop()
        print 'edge', edge
        # We have edge how do we link to others? Avoid vertex
        branch_coded = branch_from(edge, vertex, e2v, v2e, terminals)
        # Decode branch and update data structures
        branch = []
        for flip, index in branch_coded:
            edge = e2v.pop(index)
            print index, '->', (flip, edge)
            if flip:
                branch.append(edge[::-1])
            else:
                branch.append(edge)
        # Pop start_edges! so that we don't walk back etc
        if index in start_edges: start_edges.remove(index)

        print 'remain', start_edges, len(start_edges)
        print
        
        branches.append(branch)

    return branches


def branch_from(edge, avoid_vertex, e2v, v2e, terminals):
    '''List of edges until terminal is encountered'''
    v0, v1 = e2v[edge]

    flip = v1 == avoid_vertex
    next_v = v0 if flip else v1
    
    flip_branch = [(flip, edge)]

    while next_v not in terminals:
        print edge, '->', next_v, v2e(next_v)
        try:
            edge, = [e for e in v2e(next_v) if e != edge]
        except ValueError:
            break

        v0, v1 = e2v[edge]
        flip = v1 == next_v
        
        next_v = v0 if flip else v1
        flip_branch.append((flip, edge))
        print '\t', flip_branch
    return flip_branch
                    
# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitSquareMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary)
    from xii import EmbeddedMesh

    mesh = UnitSquareMesh(4, 4)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)
    
    mesh = EmbeddedMesh(f, 1)

    print len(find_branches(mesh))
