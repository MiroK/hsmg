from collections import deque
from itertools import ifilter
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

def pair(a, b):
    '''Sorted tuple'''
    return (a, b) if a < b else (b, a)

def find_segments(mesh, TOL=1E-13):
    '''Produce list of edges that represent a straight segment'''
    # Precompute connectivity, tangent vectors
    mesh.init(0, 1)
    e2v = mesh.topology()(1, 0)
    v2e = mesh.topology()(0, 1)
    # Vertex to vertex connection
    v2v = {v: set(sum((list(e2v(e)) for e in v2e(v)), [])) - set((v, ))
           for v in range(mesh.num_vertices())}

    x = mesh.coordinates()
    # Represent edgees as sorted pairs(for lookup)
    edges = [pair(*e2v(e)) for e in range(mesh.num_entities(1))]
    tangents = [x[e[0]] - x[e[1]] for e in edges]
    
    edges = {e: t/np.linalg.norm(t) for (e, t) in zip(edges, tangents)}
    
    segments = []
    while edges:
        edge, tangent = edges.popitem()
        # Grow the segment if possible by connection same oriented
        segment = segment_from(edge, tangent, v2v, edges, TOL)
        segments.append(segment)

        # Wonh be visited anymore
        for e in ifilter(lambda e: e!= edge, segment):
            del edges[e]
    return segments


def segment_from(edge, tv, v2v, edges, TOL):
    '''Grow straight segment aligned and connected to edge'''
    segment = deque([edge])
    # Possible starts
    next_edge = [edge]
    while next_edge:
        edge = next_edge.pop()
        # Check the two connections
        for vertex in edge:
            # Check only edges that are active are not in segment already
            connected_edges = ifilter(lambda e: e in edges and e not in segment,
                                          (pair(vertex, o) for o in v2v[vertex]))
            # Check all
            for e in connected_edges:
                t = edges[e]
                ip = np.dot(tv, t)
                # -->, --> 
                if abs(ip - 1) < TOL:
                    segment.append(e)
                    # A good edge is a possible start
                    next_edge.append(e)
                # -->, <---
                elif abs(ip + 1) < TOL:
                    segment.appendleft(e)
                    next_edge.append(e)

        # Break when exhausted
        # NOTE: avoiding recursion for speed (at least that is the hope)
    return list(segment)
                    
# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitSquareMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary)
    from xii import EmbeddedMesh

    mesh = UnitSquareMesh(10, 10)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)
    

    mesh = EmbeddedMesh(f, 1)

    print map(len, find_segments(mesh))
