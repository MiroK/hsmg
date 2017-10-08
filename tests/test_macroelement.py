from hsmg.macro_element import macro_dofmap
from hsmg.restriction import Dirichlet_dofs
from dolfin import UnitIntervalMesh, FunctionSpace, DomainBoundary
from dolfin import Point, vertices, near, CompiledSubDomain


def test_P1_1():
    mesh = UnitIntervalMesh(5)
    V = FunctionSpace(mesh, 'CG', 1)
    size = 1
    # Identity for P1 and size 1
    assert all(dof == v[0] for dof, v in enumerate(macro_dofmap(size, V, mesh)))


def test_P2_1():
    mesh = UnitIntervalMesh(5)
    V = FunctionSpace(mesh, 'CG', 2)
    x = V.tabulate_dof_coordinates().reshape((V.dim(), -1))

    def is_vertex_dof(dof):
        return any(near(Point(x[dof]).distance(vertex.point()), 0)
                   for vertex in vertices(mesh))

    bdry_dofs = Dirichlet_dofs(V, DomainBoundary(), [mesh])[0]
    
    size = 1
    # x-x-(x)-x-(x)-x_x
    # Bdry has 2, vertex has 3, interior is 1 (identity)
    for dof, v in enumerate(macro_dofmap(size, V, mesh)):
        if dof in bdry_dofs:
            assert len(v) == 2
        else:
            if is_vertex_dof(dof):
                assert len(v) == 3
            else:
                assert v[0] == dof


def test_P1_2():
    mesh = UnitIntervalMesh(5)
    V = FunctionSpace(mesh, 'CG', 1)
    bdry_dofs = Dirichlet_dofs(V, DomainBoundary(), [mesh])[0]
    size = 2
    for dof, v in enumerate(macro_dofmap(size, V, mesh)):
        assert (dof in bdry_dofs and len(v) == 2) or len(v) == 3


def test_P3_3():
    mesh = UnitIntervalMesh(8)
    V = FunctionSpace(mesh, 'CG', 1)
    bdry_dofs = Dirichlet_dofs(V, DomainBoundary(), [mesh])[0]
    size = 3

    count5, count4= 0, 0
    for dof, v in enumerate(macro_dofmap(size, V, mesh)):
        if dof in bdry_dofs:
            assert len(v) == 3
        else:
            assert len(v) in (5, 4)
            count5 += len(v) == 5
            count4 += len(v) == 4
    assert count5 == 5
    assert count4 == 2
