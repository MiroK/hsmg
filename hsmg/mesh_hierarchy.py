from dolfin import MeshFunction, refine
from hierarchy_1d import coarsen_1d_mesh
from hierarchy_2d import coarsen_2d_mesh


def mesh_hiearchy(mesh, by, nlevels):
    '''
    A list of nlevels+1(at most) meshes (finest first) constructed using by
    on mesh.
    '''
    assert nlevels >= 0
    assert by in ('refine', 'coarsen')

    hierarchy = [mesh]
    # Do some work
    if by == 'refine':
        while nlevels > 0:
            mesh_ = hierarchy[-1]
            where = MeshFunction('bool', mesh_, mesh_.topology().dim(), True)
            fmesh = refine(mesh_, where)
        
            hierarchy.append(fmesh_)
            nlevels -= 1
        # Coarse first
        return hierarchy[::-1]
    
    cmesh, success = hierarchy[-1], True
    # While refinement can go forever can stop earlier so we check for
    # early termination
    while nlevels > 0 and success:
        cmesh, success = coarsen(cmesh)
        
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

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitSquareMesh
    
    mesh = UnitSquareMesh(4, 4)
    print len(mesh_hiearchy(mesh, by='coarsen', nlevels=4))
