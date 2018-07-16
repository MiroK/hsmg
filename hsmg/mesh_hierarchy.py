from dolfin import MeshFunction, refine
from hierarchy_1d import coarsen_1d_mesh
from hierarchy_2d import coarsen_2d_mesh
import numpy as np


def mesh_hierarchy(mesh, nlevels):
    '''
    A list of nlevels+1(at most) meshes (finest first) obtained by coarsening
    mesh
    '''
    assert nlevels >= 0

    hierarchy = [mesh]
    
    cmesh, success = hierarchy[-1], True
    # While refinement can go forever can stop earlier so we check for
    # early termination
    while nlevels > 0 and success:
        cmesh, success, color_f = coarsen(cmesh)
        
        success and hierarchy.append(cmesh)
        nlevels -= 1
    return hierarchy


def coarsen(mesh):
    '''Coarsen manifold meshes'''
    tdim =  mesh.topology().dim()
    assert mesh.geometry().dim() > tdim > 0  # Manifold
    assert tdim < 3  # In case space time elements ...

    # Dispatch
    return {2: coarsen_2d_mesh, 1: coarsen_1d_mesh}[tdim](mesh)


def is_nested(hierarchy, TOL=1E-13):
    '''Vertices of coarse are in finer'''
    if len(hierarchy) == 1: return False
    # Recurse
    if len(hierarchy) > 2:
        return is_nested(hierarchy[:2]) and is_nested(hierarchy[1:])

    fine, coarse = hierarchy

    x = fine.coordinates()
    ys = list(coarse.coordinates())
    while ys:
        y = ys.pop()
        # Not found
        dist = np.sqrt(np.sum((x-y)**2, axis=1))
        i = np.argmin(dist)

        if dist[i] > TOL: return False
        # Chop the found one
        x = np.row_stack([x[:i], x[i+1:]])

    return True
