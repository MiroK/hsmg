from dolfin import Cell, Mesh, MeshFunction

from itertools import dropwhile, chain
from collections import namedtuple
from mesh_write import make_mesh
import subprocess, os
import numpy as np


Manifold = namedtuple('manifold', ['nodes', 'boundary'])
GeoData = namedtuple('geodata', ['points', 'lines', 'polygons'])


class GmshCoarsener(object):
    def __init__(self, path):
        root, geo = os.path.splitext(path)
        assert geo == '.geo'

        self.path = path
        self.root = root 
        self.geo_computed = False

    def coarsen(self, mesh):
        if not self.geo_computed:
            self.path = geo_file(mesh, self.path)
        # We want this size globally
        hmin = 2*mesh.hmin()

        # Generation
        subprocess.call(['gmsh -2 -setnumber size %g %s' % (hmin, self.path)],
                        shell=True)

        # All exists
        msh_file = '.'.join([self.root, 'msh'])
        assert os.path.exists(msh_file)

        # Package for dolfin
        vertices, cells, cell_markers = GmshCoarsener.parse_msh(msh_file)
        cmesh = make_mesh(vertices, cells, tdim=2, gdim=3)
        color_f = MeshFunction('size_t', cmesh, 2, 0)
        color_f.array()[:] += cell_markers

        success = cmesh.hmin() > mesh.hmin()

        return  cmesh, success, color_f

    @staticmethod
    def parse_msh(msh_file):
        vertices = []
        cells = []
        markers = []
        
        with open(msh_file, 'r') as f:
            # Look for node tag
            _ = next(dropwhile(lambda l: not l.startswith('$Nodes'), f))
            
            # Fill the vertices
            nvertices = int(next(f).strip())

            for _ in range(nvertices):
                line = next(f).strip().split()[1:]  # Ignore index
                assert len(line) == 3  # We want 3d
                vertices.append(map(float, line))

            # Look for element node tag
            _ = next(dropwhile(lambda l: not l.startswith('$Elements'), f))
            nelements = int(next(f).strip())

            # Fill cell info
            for _ in range(nelements):
                line = next(f).strip().split()[1:]  # Ignore index
                eltype, _, marker, _, v0, v1, v2 = map(int, line)
                assert eltype == 2
                cells.append((v0-1, v1-1, v2-1))
                markers.append(marker)

            assert next(f).strip() == '$EndElements'

        vertices = np.array(vertices, dtype=float)
        cells = np.array(cells, dtype='uintp')
        markers = np.array(markers, dtype='uintp')
        return vertices, cells, markers


def geo_file(mesh, path):
    '''Reconstuct geometry definition from mesh'''
    data = geo_file_data(mesh)

    with open(path, 'w') as f:
        # Write vertices
        f.write('mesh_size = 1;\n')  # To tune from outside
        point = 'Point(%d) = {%.16f, %.16f, %.16f, size};\n'
        for i, xi in enumerate(data.points, 1):
            f.write(point % ((i, ) + tuple(xi)))
        f.write('\n')

        # Write lines
        line = 'Line(%d) = {%d, %d};\n'
        for i, (v0, v1) in enumerate(data.lines, 1):
            f.write(line % (i, v0+1, v1+1))
        f.write('\n')

        # Loops
        for i, loop in enumerate(data.polygons, 1):
            loop_lines = [-(index+1) if flip else index+1 for (index, flip) in loop]
            geo_loop = 'Curve Loop(%d) = {%s};\n' % ((i, ', '.join(map(str, loop_lines))))
            f.write(geo_loop);
            f.write('Plane Surface(%d) = {%d};\n' % (i, i))
            f.write('Physical Surface(%d) = {%d};\n\n' % (i, i))

    return path
            

        
def geo_file_data(mesh):
    '''Compute data for geofile of mesh'''
    manifolds = smooth_manifolds(mesh)
    
    polygons = []  # That form boundaries of planes
    for manifold in manifolds:
        for plane in break_to_planes(manifold, mesh):
            polygons.append(plane_boundary(plane, mesh))

    # Establish global numbering of vertices the definine boundaries
    mapping = {}
    for vertex in chain(*polygons):
        index = mapping.get(vertex, len(mapping))
        mapping[vertex] = index
    # Translate
    for i in range(len(polygons)):
        polygons[i] = map(mapping.__getitem__, polygons[i])

    lines = []  # Computing global index
    polygons_lines = []
    for polygon in polygons:
        polygon_lines = []
        for v0, v1 in zip(polygon[:-1], polygon[1:]):
            if (v0, v1) in lines:
                index = lines.index((v0, v1))
                flip = False

            elif (v1, v0) in lines:
                index = lines.index((v1, v0))
                flip = True

            else:
                index = len(lines)
                lines.append((v0, v1))
                flip = False
            polygon_lines.append((index, flip))
        polygons_lines.append(polygon_lines)

    x = mesh.coordinates()
    points = np.zeros((len(mapping), x.shape[1]))
    for gi, li in mapping.items():
        points[li] = x[gi]

    return GeoData(points, lines, polygons_lines)


def plane_boundary(manifold, mesh, TOL=1E-13):
    '''
    Given a plane manifold we are after the vertex indices where the 
    condequent paits for the segments that bound it.
    '''
    mesh.init(1)
    mesh.init(1, 0)
    mesh.init(0, 1)

    # Mappings for the general algorithm
    nodes = manifold.boundary  # Mesh edges, edges in the graph are vertices
    node2edges = lambda n, n2e=mesh.topology()(1, 0): set(n2e(n))
    edge2nodes = lambda e, e2n=mesh.topology()(0, 1): set(e2n(e))

    x = mesh.coordinates()
    tangents = {n: np.diff(x[list(node2edges(n))], axis=0).flatten() for n in range(mesh.num_entities(1))}
    for node in tangents:
        tangents[node] /= np.linalg.norm(tangents[node])

    def edge_is_smooth(e, nodes=nodes, e2n=edge2nodes, tangents=tangents):
        # Vertex can be connected to many edges of the mesh
        edges = edge2nodes(e) & nodes  
        if len(edges) != 2:
            return False
        
        e0, e1 = edges
        return abs(abs(tangents[e0].dot(tangents[e1])) - 1) < TOL
        
    segments = find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)
    # Only work with boundary/vertices for now
    segments = [list(segment.boundary) for segment in segments]
    # Let's orient it - it must be a closed look
    seg = segments.pop()
    assert len(seg) == 2

    loop = seg
    v = loop[-1]
    while segments:
        # Look for segment that can be connected
        seg = next(dropwhile(lambda seg, v=v: v not in seg, segments))
        segments.remove(seg)

        v0, v1 = list(seg)
        v = v1 if v == v0 else v0
        loop.append(v)
    assert loop[0] == loop[-1]

    return loop

    
def break_to_planes(manifold, mesh, TOL=1E-13):
    '''
    Break manifold intro planes by smoothness understood in the sense 
    that the normal of 2 cells does not change orientation
    '''
    tdim = mesh.topology().dim()

    mesh.init(tdim-1)
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)

    # Mappings for the general algorithm
    nodes = manifold.nodes
    node2edges = lambda n, n2e=mesh.topology()(tdim, tdim-1): set(n2e(n))
    edge2nodes = lambda e, e2n=mesh.topology()(tdim-1, tdim): set(e2n(e))

    normals = {c: Cell(mesh, c).cell_normal() for c in range(mesh.num_cells())}

    def edge_is_smooth(e, e2n=edge2nodes, normals=normals):
        cells = edge2nodes(e)
        if len(cells) != 2:
            return False
        
        c0, c1 = cells
        return abs(abs(normals[c0].dot(normals[c1])) - 1) < TOL
        
    return find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)
    


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
    nodes = set(range(mesh.num_entities(tdim)))
    node2edges = lambda n, n2e=mesh.topology()(tdim, tdim-1): set(n2e(n))
    edge2nodes = lambda e, e2n=mesh.topology()(tdim-1, tdim): set(e2n(e))
    
    edge_is_smooth = lambda e, e2n=edge2nodes: len(e2n(e)) == 2

    return find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)


def find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth):
    '''
    Let there be a graph with nodes connected by edges. A smooth manifold 
    is a collection of nodes connected together by smooth edges.

    INPUT
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


# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary, File, cells)
    from xii import EmbeddedMesh
    from itertools import chain

    mesh = UnitCubeMesh(8, 8, 8)


    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(f, 1)
    # CompiledSubDomain('near(x[0], 0)').mark(f, 0)
    # CompiledSubDomain('near(x[0], 1)').mark(f, 0)

    # CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0.0)').mark(f, 1)
    # CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    # CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)

    # CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0)').mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    #CompiledSubDomain('near(x[0], 1.)').mark(f, 1)


    mesh = EmbeddedMesh(f, 1)

    manifolds = smooth_manifolds(mesh)

    f = MeshFunction('size_t', mesh, 2, 0)
    values = f.array()
    for c, manifold in enumerate(manifolds, 1):
        values[list(manifold.nodes)] = c

    File('gmsh_2d.pvd') << f

    # f = MeshFunction('size_t', mesh, 2, 0)
    # values = f.array()

    # color = 0
    # for manifold in manifolds:
    #     print 'manif', manifold
    #     for plane in break_to_planes(manifold, mesh):
    #         color += 1
    #         values[list(plane.nodes)] = color
    #         print '\tplane', plane
    #         plane_boundary(plane, mesh)
    # File('gmsh_2d_planes.pvd') << f

    # geo_file_data(mesh)

    # geo_file(mesh, path='test.geo')

    coarsener = GmshCoarsener('test.geo')
    _, success, after =coarsener.coarsen(mesh)

    print success
    File('after.pvd') << after

