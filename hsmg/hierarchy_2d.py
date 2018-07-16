# (Unsurprisingly) there is a lot of similarity between coarsening 2d
# and 1d: the role of branch is taken by smooth manifold
#         the role of segment is taken by plane
#         (diff) for coarsening we rely on pragmatic
#         when stitching/merging  we rely on tdim-1 boundaries
#
# The reason why I don't go for generic algorithm templated over D is
# that in 1d some things are made simpler by walking the branch/segment.
# For 2d such path (Eulerian) might not even exist

from itertools import repeat, chain
from mesh_write import make_mesh
from functools import partial
import numpy as np
# We rely on pragmatic for coarsening https://github.com/meshadaptation/pragmatic
from adaptivity import refine_metric, adapt, mesh_metric

from dolfin import MeshFunction, Cell, SubMesh


def coarsen_2d_mesh():
    pass

def mesh_from_planes(planes, TOL, debug):
    '''TODO'''
    tdim = planes[0][0].topology().dim()
    gdim = planes[0][0].geometry().dim()
    assert tdim == 2 and gdim == 3
    
    # Establish first the global(in the final) mesh numbering of shared vertices
    vertices = None  # We put in shared first
    local_to_global_maps = []
    offsets = []  # We aim to color the planes in the final mesh
    for plane in chain(*planes):
        tdim = plane.topology().dim()
        # The planes name the cells as they come so keep track of continuous
        # chunks
        offsets.append(plane.num_cells())

        # Who is a boundary facet
        facet_f = MeshFunction('size_t', plane, tdim-1, 0)
        DomainBoundary().mark(facet_f, 1)

        # Need its vertices
        plane.init(tdim-1, 0)
        f2v = plane.topology()(tdim-1, 0)

        bdry_vertices = np.hstack(map(f2v, np.where(facet_f.array() == 1)[0]))
        # Unique
        bdry_vertices = list(set(bdry_vertices))
        
        print '>>', bdry_vertices
        x = plane.coordinates()
        if vertices is None:
            vertices = x[bdry_vertices]
            lg_map = dict(zip(bdry_vertices, range(len(bdry_vertices))))
        else:
            # Try to look up each boundry vertex
            x = x[bdry_vertices]

            lg_map = {}
            new_vertices = []
            for local_i, xi in zip(bdry_vertices, x):
                dist = np.sqrt(np.sum((vertices - xi)**2, 1))
                global_i = np.argmin(dist)
                # It is new
                if dist[global_i] > TOL:
                    global_i = len(vertices)  + len(new_vertices)  # We add and this is the index
                    new_vertices.append(xi)
                    print 'adding', xi, 'as', global_i
                else:
                    print 'found', xi, 'as', global_i
                    
                lg_map[local_i] = global_i
            vertices = np.row_stack([vertices, np.array(new_vertices)])
        print vertices
        local_to_global_maps.append(lg_map)
    print local_to_global_maps
    
    # Each plane can now add the unshared vertices
    for plane, lg_map in zip(chain(*planes), local_to_global_maps):
        unshared_i = list(set(range(plane.num_vertices())) - set(lg_map.keys()))
        # Update the mapping - we know how many coords to add
        nv = len(vertices)   # First global index
        lg_map.update(dict(zip(unshared_i, range(nv, nv+len(unshared_i)))))
        # Actually add those vertices
        vertices = np.row_stack([vertices, plane.coordinates()[unshared_i]])
        print 'Inner', unshared_i
        print 'inner x', plane.coordinates()[unshared_i]
        print
    # It remains to compute the cells of the global mesh
    cells = []
    for plane, lg_map in zip(chain(*planes), local_to_global_maps):
        new = map(lg_map.__getitem__, plane.cells().flat)
        print lg_map
        for row0, row1 in zip(plane.cells(), np.array(new).reshape((-1, 3))):
            print row0, '->', row1
        print
        cells.extend(new)
    cells = np.array(cells, dtype='uintp').reshape((-1, 3))
    print cells
    # And make the global coarse mesh
    mesh =  make_mesh(vertices, cells, tdim, gdim)

    if not debug:
        return mesh

    # Plane marking cell function
    cell_f = MeshFunction('size_t', mesh, tdim, 0)
    values = cell_f.array()

    offsets = np.r_[0, np.cumsum(offsets)]
    for color, (first, last) in enumerate(zip(offsets[:-1], offsets[1:]), 1):
        values[first:last] = color

    return mesh, cell_f


def pragmatic_coarsen(mesh):
    '''Using pragmatic to coarse mesh'''
    # FIXME: explore more
    Mp = refine_metric(mesh_metric(mesh), 0.5)
    return adapt(Mp)


def coarsen_plane(plane, mesh, doit):
    '''Coarsen a plane coming here as a collection of cells(indices) of mesh'''
    # Represent plane as its own mesh (2d in 3d)
    tdim = mesh.topology().dim()
    assert tdim == 2
    
    f = MeshFunction('size_t', mesh, tdim, 0)
    values = f.array()
    values[list(plane)] = 1

    plane = SubMesh(mesh, f, 1)
    
    # Project to 2d
    n = Cell(plane, 0).cell_normal().array()
    n /= np.linalg.norm(n)
    # Define first an orthogonal system
    t1 = np.array((n[1]-n[2], n[2]-n[0], n[0]-n[1]))
    t1 /= np.linalg.norm(t1)
    # Final tangent is orthogonal to the last two guys
    t2 = np.cross(n, t1)
    t2 == np.linalg.norm(t2)
    # In this system I want coordinates of mesh.
    x3d = plane.coordinates()
    origin = x3d[0].copy()
    # Origin shift
    x3d -= origin

    # The new coordinates; FIXME: vectorize
    x2d = np.c_[np.fromiter((t1.dot(x) for x in x3d), dtype=float),
                np.fromiter((t2.dot(x) for x in x3d), dtype=float)]
    # Cells are the same
    cells = np.array(plane.cells(), dtype='uintp')

    # # Now get the 2d mesh for pragmatic
    mesh2d = make_mesh(x2d, cells, tdim, tdim)
    # Coarse by pragmatic -> get out the coordinates and cells
    cmesh, _ = doit(mesh2d)

    # Project back; FIXNE: vectorize
    y3d = np.array([t1*y[0] + t2*y[1] for y in cmesh.coordinates()])
    y3d += origin

    ycells = np.array(cmesh.cells(), dtype='uintp')

    # Tadaa
    cmesh = make_mesh(y3d, ycells, tdim, tdim+1)
    
    return cmesh
    

def find_planes(mesh, TOL):
    '''TODO'''
    smooth_manifolds = find_smooth_manifolds(mesh)
    return [break_to_planes(sm, mesh, TOL) for sm in smooth_manifolds]


def break_to_planes(smooth_manifold, mesh, TOL):
    '''A non self-intersecting manifold is broken to planes'''
    # The idea is to grow the plane as a manifold linking the cells over
    # facets which do not produce change in orientation
    x = mesh.coordinates()
    normals = {c: Cell(mesh, c).cell_normal() for c in smooth_manifold}

    tdim = mesh.topology().dim()
    mesh.init(tdim-1)
    f2c = mesh.topology()(tdim-1, tdim)
    c2f = mesh.topology()(tdim, tdim-1)
    
    planes = []
    # FIXME: here you grow from cell, manifolds were grown from vertices
    # But the difference seems to be only the notion of smoothness
    # geometric vs topological?
    while smooth_manifold:
        # Seed a new plane
        cell = smooth_manifold.pop()
        plane_normal = normals[cell]

        plane = set((cell, ))
        next_cells = [cell]
        # Grow it
        while next_cells:
            next_cell = next_cells.pop()
            # Get all the neighbors of next_cell that we can add
            neighbors = set(sum((list(f2c(f)) for f in c2f(next_cell)), []))
            # Some of them we've seen
            neighbors.difference_update(plane)
            neighbors.difference_update((next_cell, ))
            # We're only concerned with neigbors in the smooth manifold
            neighbors.intersection_update(smooth_manifold)
            # And we only grow by those that have the same orientation
            for c in neighbors:
                if abs(abs(plane_normal.dot(normals[c]))-1) < TOL:
                    plane.add(c)
                    next_cells.append(c)
        # No cell is present in more than one plane
        smooth_manifold.difference_update(plane)

        planes.append(plane)
    return planes


def find_smooth_manifolds(mesh):
    '''
    A smooth manifold is bounded by facets that are either connected to 1 
    or more than 2 cells.
    '''
    tdim = mesh.topology().dim()

    mesh.init(tdim)
    mesh.init(tdim-1)
    # Generic; facet-cell connectivity
    mesh.init(tdim, tdim-1)
    mesh.init(tdim-1, tdim)
    c2f = mesh.topology()(tdim, tdim-1)
    f2c = mesh.topology()(tdim-1, tdim)

    starts = []
    # Let's find the start facets
    for f in range(mesh.num_entities(tdim-1)):
        # Bdry of shared
        nc = len(f2c(f))
        if nc == 1 or nc > 2:
            starts.extend(zip(set(f2c(f)), repeat(f)))
            
    # We might have a single loop, then the start does not matter
    if not starts:
        # First cell and one it's facet
        starts= [(0, c2f(0)[0])]

    # Facets for termination checking
    terminals = set(p[1] for p in starts)  # Unique

    manifolds = []
    while starts:
        cell, facet = starts.pop()

        manifold = manifold_from(cell, facet, c2f, f2c, terminals)
        # Burn bridges every cell is part of only one manifold so those
        # seeds that have it
        for cell in manifold:

            remove = filter(lambda p: p[0] == cell, starts)
            for r in remove:
                starts.remove(r)
        
        manifolds.append(manifold)
    return manifolds


def manifold_from(cell, avoid_facet, c2f, f2c, terminals):
    '''List of cells bounded by terminal facets'''
    # We connect with others over non-terminal facets
    next_facets = set(c2f(cell)) - set((avoid_facet, ))
    next_facets.difference_update(terminals)
    
    manifold = set((cell, ))
    while next_facets:
        next_f = next_facets.pop()
        # This is at most one cell
        cell_of_f = set(f2c(next_f)) - manifold
        # If we run into corner we try extending by using other facet
        if not cell_of_f: continue

        c,  = cell_of_f
        manifold.add(c)
        # The connected cell may contribute new starting facets
        new_facets = set(c2f(c))
        new_facets.difference_update((next_f, ))  # Don't go back
        # We're really interested only in those that can be links
        new_facets.difference_update(terminals)
        # Exentedind the possble seeds for next round
        next_facets.update(new_facets)
    return manifold

# -------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import (UnitCubeMesh, BoundaryMesh, MeshFunction, CompiledSubDomain,
                        DomainBoundary, File, cells)
    from xii import EmbeddedMesh
    from itertools import chain

    mesh = UnitCubeMesh(8, 8, 8)


    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    # DomainBoundary().mark(f, 1)
    # CompiledSubDomain('near(x[0], 0)').mark(f, 0)
    # CompiledSubDomain('near(x[0], 1)').mark(f, 0)

    CompiledSubDomain('near(x[1], 0.0)').mark(f, 1)
    # CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    # CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)

    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0)').mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    #CompiledSubDomain('near(x[0], 1.)').mark(f, 1)

    
    mesh = EmbeddedMesh(f, 1)
    manifold_planes = find_planes(mesh, 1E-13)
    cplanes = [[coarsen_plane(p, mesh, doit=pragmatic_coarsen) for p in manifold]
               for manifold in manifold_planes]
    cmesh, after = mesh_from_planes(cplanes, 1E-13, True)

    before = MeshFunction('size_t', mesh, 2, 0)
    for c, m in enumerate(chain(*manifold_planes), 1):
        for i in m:
            before[int(i)] = c

    File('before.pvd') << before

    File('after.pvd') << after
