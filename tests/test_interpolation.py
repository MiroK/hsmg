from hsmg.restriction import interpolation_mat
from dolfin import *
import numpy as np
import pytest


@pytest.mark.parametrize('elm', [FiniteElement('Lagrange', triangle, 1),
                                 FiniteElement('Lagrange', triangle, 2),
                                 VectorElement('Lagrange', triangle, 1),
                                 VectorElement('Lagrange', triangle, 2),
                                 TensorElement('Lagrange', triangle, 1),
                                 TensorElement('Lagrange', triangle, 2),
                                 FiniteElement('Raviart-Thomas', triangle, 1),
                                 FiniteElement('Brezzi-Douglas-Marini', triangle, 1),
                                 FiniteElement('Crouzeix-Raviart', triangle, 1)])
def test_identity_property(elm):
    '''Resticting V to V is realized by identity matrix'''
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, elm)
    
    R = interpolation_mat((V, V))
    x = as_backend_type(Function(V).vector()).vec()
    Rx = x.copy()

    x.array[:] = np.random.rand(V.dim())
    R.mult(x, Rx)
        
    Rx.axpy(-1, x)
    e = Rx.norm(2)

    assert e < 1E-14*V.dim()

    
@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('degree_rise', [0, 1])
@pytest.mark.parametrize('cell', [triangle, tetrahedron])
def test_injection_property(rank, degree_rise, cell):
    '''
    Check that if VH, Vh can represent f exactly then restriction
    injects dofs if Vh to VH.
    '''
    # All the elements below should interpolate exactly the linear function
    if cell == triangle:
        f = {0: Expression('x[0] + 2*x[1]', degree=1),
             1: Expression(('x[0]+x[1]', 'x[0]-x[1]'), degree=1),
             2: Expression((('x[0]+x[1]', 'x[0]'),
                            ('x[0]-x[1]', '-x[1]')), degree=1)}[rank]
    else:
        assert cell == tetrahedron

        f = {0: Expression('x[0] + 2*x[1]-x[2]', degree=1),
             1: Expression(('x[0]+x[1]+x[2]',
                            'x[0]-x[1]-x[2]',
                            '-x[0]+2*x[1]-x[2]'), degree=1),
             2: Expression((('x[0]+x[1]', 'x[0]', 'x[1]'),
                            ('x[0]-x[1]', '-x[1]', 'x[2]'),
                            ('x[1]', 'x[2]', 'x[0]-x[1]')), degree=1)}[rank]
    
    if cell == triangle:
        mesh = UnitSquareMesh(16, 16)
        mesh_H = UnitSquareMesh(8, 8)
    else:
        mesh = UnitCubeMesh(6, 6, 6)
        mesh_H = UnitCubeMesh(3, 3, 3)


    FE = {0: FiniteElement, 1: VectorElement, 2: TensorElement}[rank]

    elements = [('Lagrange', 1),
                ('Lagrange', 2),
                ('Discontinuous Lagrange', 1)]

    if degree_rise == 0: elements.append(('Crouzeix-Raviart', 1))

    elements = [(FE(fam, cell, degree), FE(fam, cell, degree+degree_rise))
                for fam, degree in elements]

    if rank > 0:
        FE_ = FiniteElement if rank == 1 else VectorElement
        
        elements_ = [('Raviart-Thomas', 2),
                     ('Brezzi-Douglas-Marini', 1),
                     ('Nedelec 2nd kind H(curl)', 1)]

        elements_ = [(FE_(fam, cell, degree), FE_(fam, cell, degree+degree_rise))
                     for fam, degree in elements_]

    for elm_H, elm in elements:
        Vh = FunctionSpace(mesh, elm)
        VH = FunctionSpace(mesh_H, elm_H)    

        R = interpolation_mat((Vh, VH))
        x = as_backend_type(interpolate(f, Vh).vector()).vec()
        Rx = as_backend_type(interpolate(f, VH).vector()).vec()

        y = Rx.copy()
        R.mult(x, y)
        
        Rx.axpy(-1, y)
        e = Rx.norm(2)

        assert e < 1E-14*Vh.dim()


@pytest.mark.parametrize('cell', [triangle, tetrahedron])        
def test_injection_property_mixed(cell):
    '''
    Check for interpolation with some mixed element
    '''
    # All the elements below should interpolate exactly the linear function
    if cell == triangle:
        f = Expression(('x[0] + 2*x[1]',
                        'x[0]*x[1]',
                        'x[0]+x[1]'), degree=2)
    else:
        assert cell == tetrahedron

        f = Expression(('x[0] + 2*x[1] + x[0]*x[1]',
                        'x[0]*x[1]*x[2]',
                        'x[0] - x[2]',
                        'x[0] + x[1]'), degree=2)

    if cell == triangle:
        mesh = UnitSquareMesh(16, 16)
        mesh_H = UnitSquareMesh(8, 8)
    else:
        mesh = UnitCubeMesh(6, 6, 6)
        mesh_H = UnitCubeMesh(3, 3, 3)

    elm = MixedElement([VectorElement('Lagrange', cell, 2),
                        FiniteElement('Lagrange', cell, 1)])

    Vh = FunctionSpace(mesh, elm)
    VH = FunctionSpace(mesh_H, elm)
    
    R = interpolation_mat((Vh, VH))
    x = as_backend_type(interpolate(f, Vh).vector()).vec()
    Rx = as_backend_type(interpolate(f, VH).vector()).vec()

    y = Rx.copy()
    R.mult(x, y)
        
    Rx.axpy(-1, y)
    e = Rx.norm(2)

    assert e < 1E-14*Vh.dim()
