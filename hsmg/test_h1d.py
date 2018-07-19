from coarsen_1d import (find_branches, find_segments,
                        coarsen_segment_uniform,
                        coarsen_segment_topological,
                        coarsen_segment_iterative,
                        mesh_from_segments,
                        CurveCoarsenerIterative)

from mesh_hierarchy import mesh_hierarchy, is_nested


from dolfin import (UnitSquareMesh, DomainBoundary, MeshFunction, CompiledSubDomain,
                    cells)
from xii import EmbeddedMesh
import numpy as np


def square(n):
    mesh = UnitSquareMesh(n, n, 'crossed')
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)

    return f


def triangle(n):
    mesh = UnitSquareMesh(n, n, 'crossed')
    f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0)').mark(f, 1)
    CompiledSubDomain('near(x[1], 1-x[0])').mark(f, 1)

    return f


def T_shape(n):
    mesh = UnitSquareMesh(n, n, 'crossed')
    f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0)').mark(f, 1)
    CompiledSubDomain('near(x[1], 0.5)').mark(f, 1)

    return f


def X_shape(n):
    mesh = UnitSquareMesh(n, n, 'crossed')
    f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    CompiledSubDomain('near(x[1], 1-x[0])').mark(f, 1)

    return f


def OE_shape(n):
    mesh = UnitSquareMesh(n, n, 'crossed')
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    
    return f


def test_branch_find(n=16):
    
    counts = {
        square: 1, triangle: 1, T_shape: 3, X_shape: 4, OE_shape: 3
    }
    for domain, count in counts.items():
        f = domain(n)

        mesh = EmbeddedMesh(f, 1)
        branches = find_branches(mesh)
        assert len(branches) == count

    return True


def test_segments_find(n=16):
    
    counts = {
        square: 4, triangle: 3, T_shape: 3, X_shape: 4, OE_shape: 5
    }
    for domain, count in counts.items():
        f = domain(n)

        mesh = EmbeddedMesh(f, 1)
        segments = find_segments(mesh, 1E-13)
        assert sum(map(len, segments)) == count

    return True


def test_c_uniform():
    segment = np.array([[0., 0.],
                        [0.5, 0.],
                        [0.6, 0.],
                        [0.7, 0.],
                        [1.0, 0.]])
    true = np.array([[0., 0.],
                     [0.5, 0.],
                     [1.0, 0.]])

    csegment = coarsen_segment_uniform(segment)[0]
    assert np.linalg.norm(csegment-true) < 1E-13

    
def test_c_topological():
    segment = np.array([[0., 0.],
                        [0.5, 0.],
                        [0.6, 0.],
                        [0.7, 0.],
                        [1.0, 0.]])

    true = np.array([[0., 0.],
                     [0.6, 0.],
                     [1.0, 0.]])

    csegment = coarsen_segment_topological(segment)[0]
    assert np.linalg.norm(csegment-true) < 1E-13

    
def test_c_iterative():
    segment = np.array([[0., 0.],
                        [0.5, 0.],
                        [0.6, 0.],
                        [0.7, 0.],
                        [1.0, 0.]])

    true = np.array([[0., 0.],
                     [0.5, 0.],
                     [0.7, 0.],
                     [1.0, 0.]])

    csegment = coarsen_segment_iterative(segment)[0]
    assert np.linalg.norm(csegment-true) < 1E-13

    
def test_mesh_stitch():
    x = np.array([[0., 0.],
                  [0.5, 0.],
                  [1.0, 0.],
                  [1.0, 0.5],
                  [1.0, 1.0],
                  [0.5, 1.0],
                  [0.0, 1.0],
                  [0.0, 0.5],
                  [0.5, 0.5]])

    b0 = [[x[0], x[1], x[2]], [x[2], x[3], x[4]]]
    b1 = [[x[0], x[7], x[6]], [x[6], x[5], x[4]]]
    b2 = [[x[0], x[8], x[4]]]

    cmesh, _ = mesh_from_segments([b0, b1, b2], 1E-13)

    mesh = UnitSquareMesh(2, 2)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    mesh = EmbeddedMesh(f, 1)

    # Up to ordering these meshes are same
    x = cmesh.coordinates()
    y = mesh.coordinates()

    assert len(x) == len(y), map(len, (x, y))

    cmesh_2_mesh = np.zeros(len(y), dtype=int)
    for j, yj in enumerate(y):
        dist = np.sqrt(np.sum((x - yj)**2, 1))
        i = np.argmin(dist)
        assert dist[i] < 1E-13

        cmesh_2_mesh[i] = j

    # There exists such a mapping between cells
    cells = mesh.cells()
    mapped_ccells = np.array([cmesh_2_mesh[i] for i in cmesh.cells().flatten()])
    mapped_ccells = mapped_ccells.reshape((-1, 2))
    for j, yj in enumerate(cells):
        dist = np.sqrt(np.sum((mapped_ccells - yj)**2, 1))
        i = np.argmin(dist)
        if dist[i] > 1E-13:
            dist = np.sqrt(np.sum((mapped_ccells - yj[::-1])**2, 1))
            i = np.argmin(dist)
            assert dist[i] < 1E-13

            
def test_coarsen_fail():
    mesh = UnitSquareMesh(1, 1)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    mesh = EmbeddedMesh(f, 1)

    _, coarsened, color_f = CurveCoarsenerIterative.coarsen(mesh)
    assert not coarsened

    
def test_coarsen():
    mesh = UnitSquareMesh(16, 16)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    mesh = EmbeddedMesh(f, 1)

    nmesh, coarsened, color_f = CurveCoarsenerIterative.coarsen(mesh)
    # Succeeded
    assert coarsened
    # Preserved nbranches
    assert set(color_f.array()) == set((1, 2, 3)), set(color_f.array())
    # We have the same area
    area = sum(c.volume() for c in cells(mesh))
    narea = sum(c.volume() for c in cells(nmesh))
    assert abs(area - narea) < 1E-13
    # We have actually coarsened
    assert mesh.hmin() < nmesh.hmin()
    # We have have not shifted it
    assert np.linalg.norm(mesh.coordinates().min(axis=0) - nmesh.coordinates().min(axis=0)) < 1E-13
    assert np.linalg.norm(mesh.coordinates().max(axis=0) - nmesh.coordinates().max(axis=0)) < 1E-13

    
def test_hierarchy():
    mesh = UnitSquareMesh(16, 16)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    mesh = EmbeddedMesh(f, 1)
    # 8 4 2
    assert len(mesh_hierarchy(mesh, 3, coarsener=CurveCoarsenerIterative)) == 4


def test_hierarchy_short():
    mesh = UnitSquareMesh(16, 16)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    mesh = EmbeddedMesh(f, 1)
    # 16 8 4 2 1
    assert len(mesh_hierarchy(mesh, 6, coarsener=CurveCoarsenerIterative)) == 5


def test_hierarchy_nest():
    mesh = UnitSquareMesh(16, 16)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)
    mesh = EmbeddedMesh(f, 1)
    # 8 4 2
    hierarchy = mesh_hierarchy(mesh, 3, coarsener=CurveCoarsenerIterative)

    assert is_nested(hierarchy)

# --------------------------------------------------------------------

test_branch_find()
test_segments_find()
    
test_c_uniform()
test_c_topological()
test_c_iterative()

test_mesh_stitch()

test_coarsen_fail()    
test_coarsen()

test_hierarchy()
test_hierarchy_short()

test_hierarchy_nest()
