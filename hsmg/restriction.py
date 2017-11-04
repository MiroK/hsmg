from utils import transpose_matrix, to_csr_matrix, petsc_serial_matrix

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

    A function \phi_H\in V_H has coefficients given as LH_k(\phi_h) where
    LH_k is the k-th degree of freedom of VH.
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
        Cell(Vh.mesh(), 0).cell_normal()
        get_orientation_h = lambda cell: cell.orientation()
    except RuntimeError:
        warning('Unable to compute cell orientation. Falling back to 0.')
        get_orientation_h = lambda cell: 0

    try:
        Cell(VH.mesh(), 0).cell_normal()
        get_orientation_H = lambda cell: cell.orientation()
    except RuntimeError:
        warning('Unable to compute cell orientation. Falling back to 0.')
        get_orientation_H = lambda cell: 0


    mesh = Vh.mesh()
    tree = mesh.bounding_box_tree()
    limit = mesh.topology().size_global(mesh.topology().dim())

    # Coarse dof coordinates
    Hdmap = VH.dofmap()
    Hdofs_x = VH.tabulate_dof_coordinates().reshape((VH.dim(), -1))
    elm_H = VH.element()
    
    # Fine
    hdmap = Vh.dofmap()
    elm_h = Vh.element()

    # The is a dummy to be adjusted to represent basis functions of the fine space
    basis_function = Function(Vh)
    basis_function_coefs = as_backend_type(basis_function.vector()).vec().array
    # Rows
    visited_dofs = [False]*VH.dim()
    # Column values
    dof_values = np.zeros(elm_h.space_dimension(), dtype='double')

    with petsc_serial_matrix(VH, Vh) as mat:

        for cell_H in cells(VH.mesh()):
            dofs_H = Hdmap.cell_dofs(cell_H.index())
            # Alloc for dof definition
            vertex_coordinates_H = cell_H.get_vertex_coordinates()
            cell_orientation_H = get_orientation_H(cell_H)

            for local_H, dof_H in enumerate(dofs_H):

                if visited_dofs[dof_H]: continue
                
                visited_dofs[dof_H] = True

                # dof(local w.r.t to element) = basis_foo -> number
                degree_of_freedom = lambda f: elm_H.evaluate_dof(local_H,
                                                                 f,
                                                                 vertex_coordinates_H,
                                                                 cell_orientation_H,
                                                                 cell_H)
                
                # Extract coordinate for computing h collision
                x = Hdofs_x[dof_H]
        
                c = tree.compute_first_entity_collision(Point(*x))

                if c >= limit:
                    warning('Dof at %r not found', x)
                    continue

                # Fine basis function on this (fine) cells
                cell_h = Cell(mesh, c)
                vertex_coordinates_h = cell_h.get_vertex_coordinates()
                cell_orientation_h = get_orientation_h(cell_h)

                # Fine indices = columns
                hdofs = hdmap.cell_dofs(c)
                for local_h, dof_h in enumerate(hdofs):
                    # Turn into basis function of this dof
                    basis_function_coefs[dof_h] = 1.
                    
                    dof_values[local_h] = degree_of_freedom(basis_function)

                    # Revert
                    basis_function_coefs[dof_h] = 0.
                # Can fill the matrix now
                col_indices = np.array(hdofs, dtype='int32')
                # Insert
                mat.setValues([dof_H], col_indices, dof_values, PETSc.InsertMode.INSERT_VALUES)
    return mat


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

        print bc.get_boundary_values().keys()
        
        bdry_dofs.append( bc.get_boundary_values().keys() )
        # FIXME: are values of interest
        # NOTE (Trygve): Removed set for easier array slicing.

    return bdry_dofs
