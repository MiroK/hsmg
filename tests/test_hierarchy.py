from hsmg.hierarchy import by_refining, by_coarsening

from dolfin import UnitIntervalMesh, cells, near
from dolfin import BoundaryMesh, UnitSquareMesh, MeshFunction
from dolfin import CompiledSubDomain, SubMesh, UnitCubeMesh
from numpy.linalg import norm
    

def check(mesh, nlevels=7):
    '''We are able to do the hierachy for line meshes'''
    gdim = mesh.geometry().dim()
    
    from_coarse = by_refining(mesh, nlevels)
    assert len(from_coarse) == nlevels

    from_fine = by_coarsening(from_coarse[0], nlevels)

    for hierarchy in (from_coarse, from_fine):
        assert all(hierarchy[i].hmin() < hierarchy[i+1].hmin()
                   for i in range(0, len(hierarchy)-1))

    for meshc, meshf in zip(from_coarse, from_fine):
        assert meshc.num_cells() == meshf.num_cells()

        # Correspondence between cells
        for ccell in cells(meshc):
            cmp = ccell.midpoint()

            found = False
            for fcell in cells(meshf):
                fmp = fcell.midpoint()
                found = near(cmp.distance(fmp), 0)
                if found: break
            assert found
            x, y = ccell.get_vertex_coordinates(), fcell.get_vertex_coordinates()
            x = x.reshape((-1, gdim))
            y = y.reshape((-1, gdim))
            # Can be permuted
            assert near(norm(x-y), 0) or near(norm(x-y[::-1]), 0), (x, y)
    return True
            
def test_Hierarchy_1d():
    mesh = UnitIntervalMesh(4)
    assert check(mesh)

    
def test_Hierarchy_2d():
    # Painful 1d in 2d        
    mesh = BoundaryMesh(UnitSquareMesh(8, 8), 'exterior')
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    CompiledSubDomain('near(x[1], 0)').mark(subdomains, 1)
    mesh = SubMesh(mesh, subdomains, 1)

    assert check(mesh)


def test_Hierarchy_3d():
    mesh = BoundaryMesh(UnitCubeMesh(8, 8, 8), 'exterior')
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    CompiledSubDomain('near(x[2], 0)').mark(subdomains, 1)
    mesh = SubMesh(mesh, subdomains, 1)
    
    mesh = BoundaryMesh(mesh, 'exterior')
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    CompiledSubDomain('near(x[1], 0)').mark(subdomains, 1)
    mesh = SubMesh(mesh, subdomains, 1)

    assert check(mesh)
