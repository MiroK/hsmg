from utils import transpose_matrix, to_csr_matrix, petsc_serial_matrix
from dolfin import FunctionSpace, Cell, Point
from dolfin import FacetFunction, Constant, DirichletBC
from itertools import izip
from petsc4py import PETSc
import numpy as np


def restriction_mat((Vh, VH)):
    '''Restrction matrix taking Vh to VH.'''
    # Only work with pair of spaces from same Lagrange elements
    assert Vh.ufl_element() == VH.ufl_element()
    assert Vh.ufl_element().family() == 'Lagrange'
    # And scalars
    assert VH.dolfin_element().value_rank() == 0

    # Want to evaluate fine space dofs at coarse points
    mesh = Vh.mesh()
    tree = mesh.bounding_box_tree()
    limit = mesh.topology().size_global(mesh.topology().dim())

    # Coarse dof coordinates
    Hdmap = VH.dofmap()
    Hdofs = VH.tabulate_dof_coordinates().reshape((VH.dim(), -1))

    # Fine
    hdmap = Vh.dofmap()
    elm = Vh.element()

    # Alloc space for dof evaluation
    space_dim = elm.space_dimension()
    basis_values = np.zeros(space_dim)

    with petsc_serial_matrix(VH, Vh) as mat:

        for x, Hdof in izip(Hdofs, xrange(VH.dim())):
            c = tree.compute_first_entity_collision(Point(*x))

            if c >= limit:
                warninig('Dof at %r not found', x)
                continue

            cell = Cell(mesh, c)
            # Now build row
            vertex_coordinates = cell.get_vertex_coordinates()
            cell_orientation = cell.orientation()
            elm.evaluate_basis_all(basis_values, x, vertex_coordinates, cell_orientation)

            hdofs = hdmap.cell_dofs(c)  # Columns

            col_values = np.array(basis_values, dtype='double')
            col_indices = np.array(hdofs, dtype='int32')
            
            # Insert (one eval per row)
            mat.setValues([Hdof], col_indices, col_values, PETSc.InsertMode.INSERT_VALUES)
    return mat


def restriction_matrix(fine_space, mesh_hierarchy, convert=to_csr_matrix):
    '''
    Restriction matrix taking FEM space on mesh_hierarchy[i]
    to (coarser space) on mesh_hierarchy[i+1].
    '''
    elm = fine_space.ufl_element()
    fem_spaces = [FunctionSpace(mesh, elm) for mesh in mesh_hierarchy]

    R = map(restriction_mat, zip(fem_spaces[:-1], fem_spaces[1:]))
    
    R = map(convert, R)
    
    return R


def Dirichlet_dofs(V, bdry, mesh_hierarchy):
    '''
    Gather dofs corrensponding to Dirichlet boundary conditions on 
    each level of the hierarchy. List of sets.
    '''
    elm = V.ufl_element()
    
    bdry_dofs = []    
    for mesh in mesh_hierarchy:
        V = FunctionSpace(mesh, elm)
        boundary = FacetFunction('size_t', mesh, 0)
        bdry.mark(boundary, 1)

        bc = DirichletBC(V, Constant(0), boundary, 1)
        bdry_dofs.append(set(bc.get_boundary_values().keys()))
        # FIXME: are values of interest
    return bdry_dofs
