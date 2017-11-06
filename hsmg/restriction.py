from utils import transpose_matrix, to_csr_matrix, petsc_serial_matrix
from macro_element import cell_patch

from dolfin import FunctionSpace, Cell, Point, warning, as_backend_type
from dolfin import FacetFunction, Constant, DirichletBC
from dolfin import Mesh, cells, Expression, Function
from dolfin import warning

from itertools import izip
from petsc4py import PETSc
import numpy as np


def interpolation_mat((Vh, VH)):
    '''
    Interpolation matrix taking Vh to VH.

    A function basis function \phi_H \in V_H has coefficients given as 
    Lh_k(\phi_H) where Lh_k is the k-th degree of freedom of Vh.
    '''
    # NOTE: this implementation is slightly slower for CG1 elements but
    # unlike that one (using element.eval_basis*) it is generic. In particular,
    # it is not limited to function spaces with dofs being point evaluations,
    # and works without any modifications for vectr/tensor valued elements.
    
    # For this to work I only make sure that function values are the same
    assert Vh.dolfin_element().value_rank() == VH.dolfin_element().value_rank()
    assert Vh.ufl_element().value_shape() == VH.ufl_element().value_shape()

    # A case of 3d-1d will wail because cell orientation (of interval)
    # will be not defined. In this case fall back is 0
    try:
        Cell(Vh.mesh(), 0).orientation()
        get_orientation_h = lambda cell: cell.orientation()
    except RuntimeError:
        warning('Unable to compute cell orientation. Falling back to 0.')
        get_orientation_h = lambda cell: 0

    try:
        Cell(VH.mesh(), 0).orientation()
        get_orientation_H = lambda cell: cell.orientation()
    except RuntimeError:
        warning('Unable to compute cell orientation. Falling back to 0.')
        get_orientation_H = lambda cell: 0

    mesh = VH.mesh()
    tree = mesh.bounding_box_tree()
    limit = mesh.topology().size_global(mesh.topology().dim())

    # Coarse dof coordinates
    Hdmap = VH.dofmap()
    elm_H = VH.element()
    
    # Fine
    hdmap = Vh.dofmap()
    elm_h = Vh.element()

    # The is a dummy to be adjusted to represent basis functions of the coarse space
    basis_function = Function(VH)
    basis_function_coefs = as_backend_type(basis_function.vector()).vec().array
    # Rows
    visited_dofs = [False]*Vh.dim()

    hdofs_x = Vh.tabulate_dof_coordinates().reshape((Vh.dim(), -1))
    # Column values
    with petsc_serial_matrix(Vh, VH) as mat:

        for cell_h in cells(Vh.mesh()):
            dofs_h = hdmap.cell_dofs(cell_h.index())
            
            dofs_x = hdofs_x[dofs_h]
            for local_h, (dof_x, dof_h) in enumerate(zip(dofs_x, dofs_h)):

                if visited_dofs[dof_h]: continue
                
                visited_dofs[dof_h] = True

                vertex_coordinates_h = cell_h.get_vertex_coordinates()
                cell_orientation_h = get_orientation_h(cell_h)

                # dof(local w.r.t to element) = basis_foo -> number
                degree_of_freedom = lambda f: elm_h.evaluate_dof(local_h,
                                                                 f,
                                                                 vertex_coordinates_h,
                                                                 cell_orientation_h,
                                                                 cell_h)
                # Colliding cells
                cs = tree.compute_entity_collisions(Point(*dof_x))

                col_indices, dof_values = [], []
                for c in cs:
                    dofs_H = Hdmap.cell_dofs(c)
                    
                    # Fine indices = columns
                    for local_H, dof_H in enumerate(dofs_H):
                        if dof_H in col_indices: continue

                        # basis_function of dof_H
                        basis_function_coefs[dof_H] = 1.

                        col_indices.append(dof_H)

                        # Evalueate coarse basis foo at fine dofs
                        dof_value = degree_of_freedom(basis_function)
                        dof_values.append(dof_value)

                        basis_function_coefs[dof_H] = 0.

                # Can fill the matrix row
                col_indices = np.array(col_indices, dtype='int32')
                dof_values = np.array(dof_values)
                
                mat.setValues([dof_h], col_indices, dof_values, PETSc.InsertMode.INSERT_VALUES)

    # Transpose
    matT = type(mat)()
    mat.transpose(matT)
    return matT


def restriction_matrix(fine_space, mesh_hierarchy, convert=to_csr_matrix):
    '''
    Restriction matrix taking FEM space on mesh_hierarchy[i]
    to (coarser space) on mesh_hierarchy[i+1].
    '''
    elm = fine_space.ufl_element()
    fem_spaces = [FunctionSpace(mesh, elm) for mesh in mesh_hierarchy]

    R = map(interpolation_mat, zip(fem_spaces[:-1], fem_spaces[1:]))
    
    R = map(convert, R)
    
    return R


def Dirichlet_dofs(V, bdry, mesh_hierarchy):
    '''
    Gather dofs corrensponding to Dirichlet boundary conditions on 
    each level of the hierarchy. List of sets.
    '''
    elm = V.ufl_element()
    # DG has no dofs assoc. with facets so different definition is required
    # FIXME: There are some other elements like this in the FEniCS jungle
    if elm.family() == 'Discontinuous Lagrange':
        bc_def = lambda V, facet_f: DirichletBC(V, Constant(0), bdry, method='pointwise')
    else:
        bc_def = lambda V, facet_f: DirichletBC(V, Constant(0), boundary, 1)
        
    bdry_dofs = []    
    for mesh in mesh_hierarchy:
        V = FunctionSpace(mesh, elm)
        boundary = FacetFunction('size_t', mesh, 0)
        bdry.mark(boundary, 1)

        bc = bc_def(V, boundary)

        bdry_dofs.append( bc.get_boundary_values().keys() )
        # FIXME: are values of interest
        # NOTE (Trygve): Removed set for easier array slicing.

    return bdry_dofs
