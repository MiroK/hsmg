from dolfin import Mesh, cells, vertices
import numpy as np
import operator

def macro_dofmap(size, space, mesh):
    '''
    TODO
    '''
    # Recurse on hierarchy
    if not isinstance(mesh, Mesh):
        hierarchy = mesh
        return [macro_dofmap(size,
                             FuncionSpace(mesh, space.ufl_element()),
                             mesh) for mesh in hierarchy]

    # The base case of single mesh
    # For now allow only scalar Lagrange elements on a line
    assert space.ufl_element().family() ==  'Lagrange'
    assert space.dolfin_element().value_rank() == 0
    assert mesh.topology().dim() == 1

    tdim = mesh.topology().dim()
    
    dm = space.dofmap()
    # dm.cell_dofs() returns all dofs on a cell. I want to learn which
    # are interior dofs == those associated with highest top dim
    interior_idx = dm.tabulate_entity_dofs(tdim, 0)
    # Exterior is the rest
    exterior_idx = np.array([i for i in range(dm.num_element_dofs(0))
                             if i not in interior_idx])

    # Connecting exterior dof to 2 cell (1d!) that share it requires
    # notion of dof to vertex map
    # Local map for the 2 dofs
    vertex_dofs = np.array([dm.tabulate_entity_dofs(0, 0),
                            dm.tabulate_entity_dofs(0, 1)])
    
    dof_to_vertex_map = {dof[0]: vertex.index()
                         for cell in cells(mesh)
                         for vertex, dof in zip(vertices(cell),
                                                dm.cell_dofs(cell.index())[vertex_dofs])}
    
    # # Cell to cell connectivity defined over vertex
    mesh.init(0, 1)
    v2c = mesh.topology()(0, 1)

    # Closure computing macro element for an exterior dof
    def macro_element(dof, size):
        '''TODO'''
        connected_cells = tuple(v2c(dof_to_vertex_map[dof]))
        elm = (connected_cells, dof)
        # Base case: 2 (1 cell supporting the macroelement of dof)
        # --  dof --
        if size == 1: return elm

        # Use the 2 boundary dofs of the elm to grow the patch
        #  new_dof -- dof -- new_dof
        patch = set([elm])
        connected_cells = list(connected_cells)  # For type consistency
        level = 1
        while level < size:
            new_connected_cells = []
            for cell in connected_cells:
                # 2L-[C2]--1L-[C1]-dof-[C3]-1L-[C4]-2R
                # Grow by computing macro element using dof on the cell
                # which is not $dof
                new_dof = set(dm.cell_dofs(cell)[exterior_idx]) - set([dof])
                new_dof = new_dof.pop()

                elm = macro_element(new_dof, 1)
                # Exlude cells already visit
                new_connected_cells.extend([c for c in elm[0]
                                            if c not in connected_cells])
                # Duplicates are accounted for also here
                patch.add(elm)
            connected_cells = new_connected_cells
            level += 1
        return list(patch)

    dof = dof_to_vertex_map.keys()[3]
        
    # Closure for computenig interior dofs of macro elements
    def interior_dofs(elm):
        '''TODO'''
        # Base case elm from macro_elem size 1
        if isinstance(elm, tuple):
            all_dofs, exterior_dofs = [], []
            # Fill relative to cell
            cells, dof = elm
            for cell in cells:
                cell_dofs = dm.cell_dofs(cell)
                exterior_cell_dofs = cell_dofs[exterior_idx]
                all_dofs.extend(cell_dofs)
                exterior_dofs.extend(exterior_cell_dofs)
            # Interior dofs of macro element are
            return set(all_dofs) - (set(exterior_dofs) - set([dof]))
        # Larger macro element are simply a union
        return reduce(operator.or_, map(interior_dofs, elm))

    macro_map = dict()
    for cell in cells(mesh):
        cell_dofs = dm.cell_dofs(cell.index())
        # Interior dof can't be visited twice
        # FIXME: regardless of size the macro element corresponding to
        # interior dof consists only of interior dofs of this cell
        for dof in cell_dofs[interior_idx]:
            macro_map[dof] = cell_dofs[interior_idx]

        # Macro element of exterior dofs vary with size
        for dof in cell_dofs[exterior_idx]:
            # Shared
            if dof in macro_map: continue
            
            macro_map[dof] = interior_dofs(macro_element(dof, size))
    # Collapse, as indices for np indexing
    macro_map = [np.fromiter(macro_map[dof], dtype=int) for dof in sorted(macro_map.keys())]
    return macro_map

from dolfin import UnitIntervalMesh, FunctionSpace

mesh = UnitIntervalMesh(3)
V = FunctionSpace(mesh, 'CG', 6)
size = 2

for k, v in enumerate(macro_dofmap(size, V, mesh)):
    print k, v
