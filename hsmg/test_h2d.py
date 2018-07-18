from coarsen_common import smooth_manifolds
from coarsen_2d import break_to_planes, plane_boundary, GmshCoarsener
from mesh_hierarchy import mesh_hierarchy

from dolfin import (UnitCubeMesh, DomainBoundary, MeshFunction, CompiledSubDomain,
                    cells)
from xii import EmbeddedMesh
import numpy as np


def Plate(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)

    return f


def Lshape(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0)').mark(f, 1)

    return f


def Oshape(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0)').mark(f, 1)
    CompiledSubDomain('near(x[0], 1)').mark(f, 1)
    CompiledSubDomain('near(x[1], 1)').mark(f, 1)

    return f


def Cube(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(f, 1)

    return f


def CubeI(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)

    return f


def CubeX(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)

    return f

    
def test_manifold_find(n=16):
    
    counts = {
        Lshape: 1, Oshape: 1, Cube: 1, CubeI: 3, CubeX: 8
    }
    for domain, count in counts.items():
        f = domain(n)
        mesh = EmbeddedMesh(f, 1)
        manifolds = smooth_manifolds(mesh)
        assert len(manifolds) == count

    return True


def test_plane_find(n=16):
    
    counts = {
        Lshape: 2, Oshape: 4, Cube: 6, CubeI: 11, CubeX: 20
    }
    for domain, count in counts.items():
        f = domain(n)
        mesh = EmbeddedMesh(f, 1)
        count_ = sum(len(break_to_planes(m, mesh)) for m in smooth_manifolds(mesh))
        assert count == count_

    return True


def test_bdry_find_edges():
    counts = {
        Lshape: [8, 8], Oshape: [8, 8, 8, 8], CubeI: [6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8]
    }
    for domain, count in counts.items():
        f = domain(n=2)
        mesh = EmbeddedMesh(f, 1)
        count_ = sum(([len(p.boundary) for p in break_to_planes(m, mesh)]
                      for m in smooth_manifolds(mesh)), [])
        assert count == sorted(count_)

    return True


def test_bdry_find_bdry():
    shapes = [Lshape, Oshape, CubeI]

    for domain in shapes:
        f = domain(n=4)
        mesh = EmbeddedMesh(f, 1)
        # We have only rectangles
        assert all(len(plane_boundary(p, mesh)) == 5
                   for m in smooth_manifolds(mesh)
                   for p in break_to_planes(m, mesh)
        )

    return True


def test_coarsening():
    mesh = UnitCubeMesh(8, 8, 8)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)

    mesh = EmbeddedMesh(f, 1)

    coarsener = GmshCoarsener('test.geo')
    cmesh, success, after = coarsener.coarsen(mesh)

    assert success
    # Preserved area
    volume = lambda mesh: sum(c.volume() for c in cells(mesh))
    assert abs(volume(mesh) - volume(cmesh)) < 1E-10
    
    # Preseverd bbox
    assert np.linalg.norm(mesh.coordinates().min(axis=0)-
                          cmesh.coordinates().min(axis=0)) < 1E-10

    assert np.linalg.norm(mesh.coordinates().max(axis=0)-
                          cmesh.coordinates().max(axis=0)) < 1E-10

    # Iscoarser
    assert cmesh.hmin() > mesh.hmin()
    
    return True


def test_not_coarsening():
    mesh = UnitCubeMesh(1, 1, 1)
    f = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    DomainBoundary().mark(f, 1)

    mesh = EmbeddedMesh(f, 1)
    coarsener = GmshCoarsener('test.geo')
    cmesh, success, after = coarsener.coarsen(mesh)

    assert not success

    
def test_hierarchy():
    f = Oshape(16)
    mesh = EmbeddedMesh(f, 1)

    assert len(mesh_hierarchy(mesh, 3, coarsener=GmshCoarsener('test.geo'))) == 4

# -------------------------------------------------------------------

test_manifold_find(4)
test_manifold_find(6)

# test_plane_find(4)
# test_plane_find(8)

# test_bdry_find_edges()
# test_bdry_find_bdry()

# test_coarsening()
# test_not_coarsening()

# test_hierarchy()

from dolfin import Timer

ns = []
times = []
for n in (2, 4, 8, 16, 32, 64):
    f = Plate(n)
    mesh = EmbeddedMesh(f, 1)
    ns.append((mesh.num_cells(), mesh.num_vertices()))
    
    t = Timer('fpp')
    smooth_manifolds(mesh)
    times.append(t.stop())
    print times[-1]

import matplotlib.pyplot as plt

n, v = map(np.array, list(zip(*ns)))

print 'n complex', np.polyfit(np.log(n), np.log(times), deg=1)[0]
print 'v complex', np.polyfit(np.log(v), np.log(times), deg=1)[0]

plt.figure()
plt.loglog(n, times, label='n')
plt.loglog(v, times, label='v')
plt.legend(loc='best')
plt.show()
