from hsmg.restriction import interpolation_mat
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from dolfin import *
import numpy as np
import pytest


def CurveIn3d(n):
    mesh = UnitCubeMesh(n, n, n)    
    f = MeshFunction('size_t', mesh, 1, 0)
    for x0, x1 in (('x[0]', 'x[1]'), ('x[1]', 'x[2]'), ('x[0]', 'x[2]')):
        for i in (0, 1):
            for j in (0, 1):
                CompiledSubDomain('near(%s, A) && near(%s, B)' % (x0, x1),
                                  A=i, B=j).mark(f, 1)
    return EmbeddedMesh(mesh, f, 1).mesh


@pytest.mark.parametrize('mesh', [lambda n: UnitIntervalMesh(n),
                                  lambda n: UnitSquareMesh(n, n),
                                  lambda n: BoundaryMesh(UnitSquareMesh(n, n), 'exterior'),
                                  lambda n: BoundaryMesh(UnitCubeMesh(n, n, n), 'exterior'),
                                  lambda n: CurveIn3d(n),
                                  lambda n: UnitCubeMesh(n, n, n)])
@pytest.mark.parametrize('nH', [2, 3, 5])
@pytest.mark.parametrize('family', ['Lagrange', 'Discontinuous Lagrange'])        
@pytest.mark.parametrize('degree', [0, 1, 2])
@pytest.mark.parametrize('scale', [2, 3, 5])
def test_sanity(mesh, nH, family, degree, scale):

    if family == 'Lagrange' and degree == 0: return True
    
    mesh_H = mesh(nH)                                                                                                                     
    VH = FunctionSpace(mesh_H, family, degree)                                                                                                               

    mesh_h = mesh(nH*scale)
    Vh = FunctionSpace(mesh_h, family, degree)                                                                                                               
    
    R = PETScMatrix(interpolation_mat((Vh, VH)))
    
    f = interpolate(Expression('2*x[0]', degree=1), VH)

    Pf = Function(Vh)
    # When coarse prolongates to fine something linear
    R.transpmult(f.vector(), Pf.vector())
    # It is the same as we interpolated on fine
    Pf0 = interpolate(f, Vh).vector()
    Pf0.axpy(-1, Pf.vector())
    
    assert Pf0.norm('linf') < 1E-14, Pf0.norm('linf')

    return True
