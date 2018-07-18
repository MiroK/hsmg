from dolfin import Cell, Mesh, MeshFunction

from itertools import dropwhile, chain
from collections import namedtuple
from mesh_write import make_mesh
import subprocess, os
import numpy as np


from coarsen_common import smooth_manifolds, find_smooth_manifolds, Manifold

GeoData = namedtuple('geodata', ['points', 'lines', 'polygons'])


class GmshCoarsener(object):
    '''Coarsen mesh by calling gmsh'''
    def __init__(self, path):
        root, ext = os.path.splitext(path)
        # Be anal about the extension
        assert ext == '.geo'

        self.path = path
        self.root = root 
        self.geo_computed = False

    def coarsen(self, mesh):
        '''
        Make an attempt at coarsening the mesh.

        INPUT:
        mesh = Mesh

        OUTPUT:
        (Mesh, success flag, MeshFunction (marking planes for debugging))

        '''
        # The expensive geometry is computed only once as the assumption
        # is that the coarsening can't introduce new polygons
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

        # Package for dolfin <- build the mesh in memory
        vertices, cells, cell_markers = GmshCoarsener.parse_msh(msh_file)
        cmesh = make_mesh(vertices, cells, tdim=2, gdim=3)
        color_f = MeshFunction('size_t', cmesh, 2, 0)
        color_f.array()[:] += cell_markers

        # Did we actually manage to coarsen it?
        success = cmesh.hmin() > mesh.hmin()

        return  cmesh, success, color_f

    @staticmethod
    def parse_msh(msh_file):
        '''Read in vertices, cells and markers from msh_file'''
        # NOTE: 3d vertices, triangular cells
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
                assert eltype == 2  # Must be triangular cells
                cells.append((v0-1, v1-1, v2-1))
                markers.append(marker)

            assert next(f).strip() == '$EndElements'

        vertices = np.array(vertices, dtype=float)
        cells = np.array(cells, dtype='uintp')
        markers = np.array(markers, dtype='uintp')
        
        return vertices, cells, markers


def geo_file(mesh, path):
    '''Write the geo file that is the recunstructed geometry of the mesh'''
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
    '''
    Compute data for geofile of mesh. We perform coarsening by a representation 
    of the geometry for gmsh and have it tessilate the domain.

    INPUT:
    mesh = Mesh (embedded)

    OUPUT:
    Geodata; domain represented as vertices, lines (defined in terms of 
    vertices), polygons (defined in terms of lines)
    '''
    manifolds = smooth_manifolds(mesh)
    
    polygons = []  # That form boundaries of planes
    for manifold in manifolds:
        for plane in break_to_planes(manifold, mesh):
            polygons.append(plane_boundary(plane, mesh))

    # Establish global numbering of vertices the definine boundaries
    mapping = {}
    for vertex in chain(*polygons):
        index = mapping.get(vertex, len(mapping))
        mapping[vertex] = index  # Mesh index to geo index
    # Translate vertices
    for i in range(len(polygons)):
        polygons[i] = map(mapping.__getitem__, polygons[i])

    # In the polygonal loop subsequent vertices form lines. Lines might
    # be shared (not necessarily with same orientation) so we need to
    # give them global index
    lines = []  # Computing global index
    polygons_lines = []
    for polygon in polygons:
        polygon_lines = []
        for v0, v1 in zip(polygon[:-1], polygon[1:]):
            # Seen with same orientation
            if (v0, v1) in lines:
                index = lines.index((v0, v1))
                flip = False
            # Seen with different orientation
            elif (v1, v0) in lines:
                index = lines.index((v1, v0))
                flip = True
            # Not seen so we get to name it
            else:
                index = len(lines)
                lines.append((v0, v1))
                flip = False
            polygon_lines.append((index, flip))
        polygons_lines.append(polygon_lines)

    # Coordinates of geo points
    x = mesh.coordinates()
    points = np.zeros((len(mapping), x.shape[1]))
    for gi, li in mapping.items():
        points[li] = x[gi]

    return GeoData(points, lines, polygons_lines)


def plane_boundary(manifold, mesh, TOL=1E-13):
    '''
    Given a plane manifold we are after the vertex indices where the consequent 
    pairs form the segments that bound it. These vertices are indices where 
    the loop that form the polygon changes orientation, i.e. tangent vectors
    have kink here

    INPUT:
    manifold = Manifold
    mesh = Mesh
    TOL = float giving tolerance for changes in tangent

    OUPUT:
    loop = [int] (mesh vertex indices)
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
        # Vertex can be connected to many edges of the mesh. Only want
        # those edges that form the boundary
        edges = edge2nodes(e) & nodes  
        if len(edges) != 2:
            return False
        
        e0, e1 = edges
        return abs(abs(tangents[e0].dot(tangents[e1])) - 1) < TOL
        
    segments = find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)
    # Only work with .boundary(i.e. physical vertices) from now on
    segments = [list(segment.boundary) for segment in segments]
    # Let's orient it - it must be a closed loop
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

    INPUT:
    manifold = Manifold
    mesh = Mesh instance
    
    OUPUT:
    list([Manifold])
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
        # No point to filter cells here as more than 2 cells indicates manifold
        # bdry which is also a plane boundary
        if len(cells) != 2:  
            return False
        
        c0, c1 = cells
        return abs(abs(normals[c0].dot(normals[c1])) - 1) < TOL
        
    return find_smooth_manifolds(nodes, node2edges, edge2nodes, edge_is_smooth)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary, File, cells)
    from xii import EmbeddedMesh
    from itertools import chain

    mesh = UnitCubeMesh(16, 16, 16)


    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(f, 1)
    #CompiledSubDomain('near(x[0], 0.)').mark(f, 1)
    #CompiledSubDomain('near(x[1], 1.)').mark(f, 1)

    # CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0.0)').mark(f, 1)
    # CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    # CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)

    # CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    #CompiledSubDomain('near(x[0], 1.)').mark(f, 1)


    mesh = EmbeddedMesh(f, 1)

    # manifolds = smooth_manifolds(mesh)

    # f = MeshFunction('size_t', mesh, 2, 0)
    # values = f.array()
    # for c, manifold in enumerate(manifolds, 1):
    #     print manifold
    #     values[list(manifold.nodes)] = c

    # File('gmsh_2d.pvd') << f

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

