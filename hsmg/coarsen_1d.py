from itertools import ifilter, repeat, dropwhile, takewhile
from coarsen_common import smooth_manifolds, find_smooth_manifolds
from collections import defaultdict
from mesh_write import make_mesh
import numpy as np

from dolfin import MeshFunction


def mesh_from_segments(segments, TOL):
    '''Create mesh given branches which contain segments'''
    # Compute first the collision between branches as they first and last
    # nodes contain the only shared nodes
    vertices = np.array([segments[0][0]])
    terminal_indices = []

    for seg in segments:
        seg_terminal_indices = []
        for x in seg[[0, -1]]:
            # Is it in?
            dist = np.sqrt(np.sum((vertices - x)**2, 1))
            i = np.argmin(dist)
            # It is not
            if dist[i] > TOL:
                i = len(vertices)  # We add and this is the index
                vertices = np.row_stack([vertices, x])
            seg_terminal_indices.append(i)
        terminal_indices.append(seg_terminal_indices)

    # Indepently
    cells = []
    color_offsets = [0]
    for (gfirst, glast), seg in zip(terminal_indices, segments):
        # Numbering for middle guys
        n = len(vertices)
        new_vertices = seg[1:-1]
        index_map = [gfirst] + range(n, n + len(new_vertices)) + [glast]
        # Cells are just conseq. pairs
        new_cells = zip(index_map[:-1], index_map[1:])
        cells.extend(new_cells)
        # Don't forget the vertices
        vertices = np.row_stack([vertices, new_vertices])
        # and cells
        color_offsets.append(color_offsets[-1] + len(new_cells))

    vertices = np.array(vertices, dtype=float)
    cells = np.array(cells, dtype='uintp')
    gdim = vertices.shape[1]
    
    mesh = make_mesh(vertices, cells, 1, gdim)

    # Color
    color_f = MeshFunction('size_t', mesh, 1, 0)
    array = color_f.array()
    for c, (f, l) in enumerate(zip(color_offsets[:-1], color_offsets[1:]), 1):
        array[f:l] = c

    return mesh, color_f
        
                     
def add_branch(segments, terminal_indices, vertices, cells):
    '''Extend vertex and cell data by adding segment vertices and cells'''
    # Note first that segments are linked so we can drop the shared pieces
    merged = segments[0]
    for seg in segments[1:]:
        merged = np.row_stack([merged, seg[1:]])

    # Now we know global index of of first and last vertices (shared ones)
    start = len(vertices)  # The first free
    vertex_indices = range(start, start+len(merged)-2)  # new guyd
    vertex_indices = [terminal_indices[0]] + vertex_indices + [terminal_indices[-1]]

    new_vertices = merged[1:-1]

    vertices.extend(new_vertices)
    cells.extend(zip(vertex_indices[:-1], vertex_indices[1:]))

    
def find_segments(mesh, TOL):
    '''FIXME'''
    # A list of manifolds
    branches = find_branches(mesh)

    mesh.init(1, 0)
    c2v = mesh.topology()(1, 0)
    
    segments = []
    for branch in branches:

        branch_segments = []
        for seg in break_to_segments(branch.nodes, mesh, TOL):
            # Want it as ordered vertices
            vfirst, vlast = seg.boundary
            # Cell vertex connectivity
            cell2vertices = {c: c2v(c) for c in seg.nodes}
            segment = [vfirst]
            while vfirst != vlast:
                # Give me the first cell connected to it
                cell = next(dropwhile(lambda c, v=vfirst: v not in cell2vertices[c], cell2vertices))
                # Get its vertices and forget
                v0, v1 = cell2vertices.pop(cell)
                # Which to link
                vfirst = v0 if v1 == vfirst else v1
                segment.append(vfirst)
            branch_segments.append(segment)
        segments.append(branch_segments)

    return segments
                
            
def break_to_segments(branch, mesh, TOL):
    '''FIXME'''
    mesh.init(1)
    mesh.init(1, 0)

    # Mappings for the general algorithm
    nodes = branch

    node2edges = mesh.topology()(1, 0)
    node2edges = {n: set(node2edges(n)) for n in nodes}
    # Invert
    edge2nodes = defaultdict(set)
    for node, edges in node2edges.iteritems():
        for e in edges:
            edge2nodes[e].add(node)

    x = mesh.coordinates()
    tangents = {n: np.diff(x[list(edges)], axis=0).flatten()
                for n, edges in node2edges.iteritems()}
    # Normalize
    for tau in tangents.itervalues():
        tau /= np.linalg.norm(tau)

    edge_is_smooth = dict(zip(edge2nodes, repeat(False)))
    for vertex, edges in edge2nodes.iteritems():
        if len(edges) != 2:
            continue

        e0, e1 = edges
        edge_is_smooth[vertex] = abs(abs(tangents[e0].dot(tangents[e1])) - 1) < TOL
        
    return find_smooth_manifolds(node2edges, edge2nodes, edge_is_smooth)


def find_branches(mesh):
    '''FIXME'''
    return smooth_manifolds(mesh)

# Some coarsening functions        

def coarsen_segment_uniform(segment):
    '''
    A segment is represented by a list of vertices where (i, i+1) 
    is an edge. Coarsening it produces a uniform mesh.
    '''
    if len(segment) == 2: return segment, False

    # The nested case okay for power of 2
    nv = len(segment)
    # Odd
    # Otherwise geometric
    A, B = segment[[0, -1]]
    dX = B - A
    segment = np.array([A + (B-A)*t for t in np.linspace(0, 1, nv/2+1)])

    return segment, True


def coarsen_segment_topological(segment):
    '''
    A segment is represented by a list of vertices where (i, i+1) 
    is an edge. Just join 2 neighboring vertices.
    '''
    if len(segment) == 2: return segment, False

    csegment = segment[::2]
    
    if len(segment) % 2 == 1: return csegment, True
    
    return np.r_[csegment, segment[-1]], True


def coarsen_segment_iterative(segment):
    '''
    A segment is represented by a list of vertices where (i, i+1) 
    is an edge. Produce mesh where hmin is doubled
    '''
    if len(segment) == 2: return segment, False

    distances = lambda seg: np.array(map(np.linalg.norm, np.diff(seg, axis=0)))

    dx = distances(segment)
    hmin = this_hmin = min(dx)

    while this_hmin < 2*hmin:
        i = np.argmin(dx[:-1] + dx[1:])
        segment = np.r_[segment[:i+1], segment[i+2:]]

        dx = distances(segment)
        this_hmin = min(dx)

    return segment, True


def coarsen_1d_mesh(mesh, coarsen_segment, TOL=1E-13):
    '''Coarse a manifold mesh of lines'''
    # The idea here is that the mesh coarsening should preserve straigth
    # segments. To this end the mesh is first broken intro straight segments
    # (ordered collection of vertices) these are then coarsened and finally
    # stiched together to form the coarse mesh
    branch_segments = find_segments(mesh, TOL)
    x = mesh.coordinates()
    # Each branch consists is a list of segments. We keep this grouping
    coarse_segments, success = [], True
    for branch in branch_segments:
        # Go over segments in this branch
        for s in branch:
            segment = x[s]
            # Work horse
            csegment, coarsened = coarsen_segment(segment)
            # Each segment must be coarsed for the final mesh
            if coarsened:
                coarse_segments.append(csegment)
            else:
                coarse_segments.append(segment)  # The same
            # Everybody needs to succeed
            success = success and coarsened
    # For debugging purposes with use cell function marking branches
    cmesh, color_f = mesh_from_segments(coarse_segments, TOL)
    
    return cmesh, success, color_f


class CurveCoarsener(object):
    def __init__(self, method):
        self.method = method

    def coarsen(self, mesh):
        return coarsen_1d_mesh(mesh, self.method)

    
CurveCoarsenerIterative = CurveCoarsener(coarsen_segment_topological)

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitSquareMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary, File, cells)
    from xii import EmbeddedMesh

    mesh = UnitSquareMesh(32, 32, 'crossed')
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)

    # CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0)').mark(f, 1)

    mesh = EmbeddedMesh(f, 1)
    print 'length', sum(c.volume() for c in cells(mesh))
    print 'hmin', mesh.hmin()
    print 
    
    cmesh, status, color_f = coarsen_1d_mesh(mesh)
    print 'length', sum(c.volume() for c in cells(cmesh))
    print 'hmin', cmesh.hmin()
    print 'nbranches', len(set(color_f.array()))

    x = MeshFunction('size_t', mesh, 1, 0)
    File('foo.pvd') << x

    File('bar.pvd') << color_f
