# (Unsurprisingly) there is a lot of similarity between coarsening 2d
# and 1d: the role of branch is taken by smooth manifold
#         the role of segment is taken by plane
#         (diff) for coarsening we rely on pragmatic
#         when stitching/merging  we rely on tdim-1 boundaries
#
# The reason why I don't go for generic algorithm templated over D is
# that in 1d some things are made simpler by walking the branch/segment.
# For 2d such path (Eulerian) might not even exist

from itertools import ifilter, repeat, dropwhile, takewhile
from functools import partial
from mesh_write import make_mesh
import numpy as np

from dolfin import MeshFunction, Cell, SubMesh


# def coarsen_2d_mesh(mesh, TOL=1E-13, debug=True):
#     '''Coarse a manifold mesh of triangles'''
#     # The idea is that a work horse is refinement of straight segments
#     # These come in grouped by branch that they come from
#     branch_segments = find_segments(mesh, TOL)
#     x = mesh.coordinates()
#     # Each branch consists is a list of segments. We keep this grouping
#     coarse_segments, success = [], False
#     for branch in branch_segments:
#         coarse_branch = []
#         # Go over segments in this branch
#         for s in branch:
#             segment = x[s]
#             # Work horse
#             csegment, coarsened = coarsen_segment(segment)
#             # Might fail but some other segment could succeed
#             if coarsened:
#                 coarse_branch.append(csegment)
#                 # so we proceeed with the original
#             else:
#                 coarse_branch.append(segment)
#             # At least one success
#             success = success or coarsened
#         # So the coarse branch is also collection of segments
#         coarse_segments.append(coarse_branch)
#     # No?
#     if not success:
#         return mesh, False
#     # Stich together
#     cmesh, color_f = mesh_from_segments(coarse_segments, TOL, debug)

#     if debug:
#         return cmesh, True, color_f
#     return cmesh, True

def coarse_plane(plane, mesh):
    '''Coarsen a plane coming here as a collection of cells(indices) of mesh'''
    # Represent plane as its own mesh (2d in 3d)
    f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    values = f.array()
    values[list(plane)] = 1

    plane = SubMesh(mesh, f, 1)

    # Project to 2d
    


    # Coarse by pragmatic


    # Project back

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

    mesh = UnitCubeMesh(2, 2, 2)


    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)
    # CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    # CompiledSubDomain('near(1-x[0], x[1])').mark(f, 1)

    # CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    # CompiledSubDomain('near(x[1], 0.0)').mark(f, 1)
    #CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    #CompiledSubDomain('near(x[0], 1.)').mark(f, 1)

    
    mesh = EmbeddedMesh(f, 1)
    mf = find_smooth_manifolds(mesh)
    for m in mf:
        len(break_to_planes(m, mesh, TOL=1E-13))
    # print 'length', sum(c.volume() for c in cells(mesh))
    # print 'hmin', mesh.hmin()
    # print 
    
    # cmesh, status, color_f = coarsen_1d_mesh(mesh)
    # print 'length', sum(c.volume() for c in cells(cmesh))
    # print 'hmin', cmesh.hmin()
    # print 'nbranches', len(set(color_f.array()))

    x = MeshFunction('size_t', mesh, 2, 0)
    for c, m in enumerate(mf, 1):
        for i in m:
            x[int(i)] = c

    File('bar.pvd') << x
