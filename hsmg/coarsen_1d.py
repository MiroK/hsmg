from itertools import ifilter, repeat, dropwhile, takewhile
from mesh_write import make_mesh
import numpy as np

from dolfin import MeshFunction


def mesh_from_segments(branch_segments, TOL):
    '''Create mesh given branches which contain segments'''
    # Compute first the collision between branches as they first and last
    # nodes contain the only shared nodes
    vertices = np.array([branch_segments[0][0][0]])
    # Get global vertex numbers
    terminal_numbers = []
    for i, bi in enumerate(branch_segments):
        branch_terminals = bi[0][0], bi[-1][-1]

        tm = ()
        for t in branch_terminals:
            # Is it in?
            dist = np.sqrt(np.sum((vertices - t)**2, 1))
            i = np.argmin(dist)
            # It is not
            if dist[i] > TOL:
                i = len(vertices)  # We add and this is the index
                vertices = np.row_stack([vertices, t])
            tm = tm + (i, )
        terminal_numbers.append(tm)
                
    # With this info each branch can extend the vertices and add cells
    # independently
    vertices, cells = list(vertices), []
    color_offsets = [0]
    for tm, branch in zip(terminal_numbers, branch_segments):
        add_branch(branch, tm, vertices, cells)  # Bang method
        # We add cells by branch (continuous chunks) so this way we keep
        # track of color
        color_offsets.append(len(cells))

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
    '''
    Produce representation of straight segments of mesh of the form 
    [[v0, ..., vn],  [vn, ...., vm]] with v_i the vertex indices
    '''
    # The idea is that any branch in the mesh must for such consist of
    # at least one straight segment. So we make branches which can be walked
    # and the walk broken once the orientation changes. This takes advantage
    # of existence and simle computation of Eulerian paths in the graph
    # and no intermediate mesh needs to be created until the final (cf 2d.)
    branches = find_branches(mesh)
    
    return [break_to_segments(branch, mesh, TOL) for branch in branches]


def break_to_segments(branch, mesh, TOL):
    '''
    We have a linked list of pairs which we break upon changes in 
    tangent orientation.
    '''
    # So now geometrical (previously topologial) smoothness matters
    x = mesh.coordinates()
    
    if len(branch) == 1: return branch
    
    # If the branch comes from one loop then the starting point was
    # chosen randomly. However, the algorithm below assumes that the
    # branch start is also a segment start. So for loops we rearange.
    def tangent(edge, x=mesh.coordinates()):
        t = x[edge[0]] - x[edge[1]]
        return t/np.linalg.norm(t)
    
    tan_change = lambda t1, t2: abs(abs(np.dot(t1, t2))-1) > TOL
    
    efirst, elast = branch[0], branch[-1]
    vfirst, vlast = efirst[0], elast[-1]
    # A loop
    if vfirst == vlast:

        edge0 = branch[0]
        t = tangent(edge0)
        # Find the first change in tangent
        i = next(dropwhile(lambda i: not tan_change(t, tangent(branch[i])), range(len(branch))))
        # Ensure that we start from the kink
        branch = branch[i:] + branch[:i]
    
    segments = []

    index = 0
    while index < len(branch):
        edge = branch[index]
        t1 = tangent(edge)
        index += 1

        segment = edge
        link = edge[1]
        # We build the segment until there is change 
        for v0, v1 in  takewhile(lambda e: not tan_change(t1, tangent(e)), branch[index:]):
            assert v0 == link   # Link prperty between edges in branch
            link = v1
            segment.append(link)
            index += 1
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
    # NOTE: I say that the manifold is smooth at facet if it is connected
    # to at most 2 cells.
    
    # The relevant connectivity
    # Precompute connectivity
    mesh.init(0, 1)
    e2v = mesh.topology()(1, 0)
    v2e = mesh.topology()(0, 1)

    starts = []  # Pair of edge and vertex to avoid
    # Let's find the start edges
    for v in range(mesh.num_vertices()):
        # Bdry point
        ne = len(v2e(v))
        if ne == 1 or ne > 2:
            starts.extend(zip(v2e(v), repeat(v)))
            
    # We might have a single loop, then the start does not matter
    if len(starts) == 0:
        # First edge, and one of its vertices
        starts.append((0, e2v(0)[0]))

    # Want to represent edge connectivity by pop-able strucure
    e2v = {e: list(e2v(e)) for e in range(mesh.num_cells())}
    # Vertices for termination checking
    terminals = set(p[1] for p in starts)  # Unique

    branches = []
    while starts:
        edge, vertex = starts.pop()
        # We have edge how do we link to others? Avoid vertex
        branch_coded = branch_from(edge, vertex, e2v, v2e, terminals)
        # Branch comes back as list of edge indices and a flag indicating
        # orientation
        # Decode branch and update data structures
        branch = []
        for flip, index in branch_coded:
            edge = e2v.pop(index)
            if flip:
                branch.append(edge[::-1])
            else:
                branch.append(edge)

        # Pop start_edges! so that we don't walk back etc
        try:
            i = next(dropwhile(lambda i: starts[i][0] != index,
                               range(len(starts))))
            del starts[i]
        except StopIteration:
            pass
        
        branches.append(branch)

    return branches


def branch_from(edge, avoid_vertex, e2v, v2e, terminals):
    '''List of edges until terminal is encountered'''
    v0, v1 = e2v[edge]

    flip = v1 == avoid_vertex
    next_v = v0 if flip else v1
    
    flip_branch = [(flip, edge)]
    while next_v not in terminals:
        # In a smooth case we have 2 connected cells and one is the previous
        # cell
        try:
            edge, = [e for e in v2e(next_v) if e != edge]
        except ValueError:
            # We've reached the boundary (or bifurcation)
            break

        v0, v1 = e2v[edge]
        flip = v1 == next_v
        
        next_v = v0 if flip else v1
        flip_branch.append((flip, edge))
    return flip_branch

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
        coarse_branch = []
        # Go over segments in this branch
        for s in branch:
            segment = x[s]
            # Work horse
            csegment, coarsened = coarsen_segment(segment)
            # Each segment must be coarsed for the final mesh
            if coarsened:
                coarse_branch.append(csegment)
            else:
                coarse_branch.append(segment)  # The same
            # Everybody needs to succeed
            success = success and coarsened
        # So the coarse branch is also collection of segments
        coarse_segments.append(coarse_branch)
    # For debugging purposes with use cell function marking branches
    cmesh, color_f = mesh_from_segments(coarse_segments, TOL)

    return cmesh, success, color_f


class CurveCoarsener(object):
    def __init__(self, method):
        self.method = method

    def coarsen(self, mesh):
        return coarsen_1d_mesh(mesh, self.method)

    
CurveCoarsenerIterative = CurveCoarsener(coarsen_segment_iterative)

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
