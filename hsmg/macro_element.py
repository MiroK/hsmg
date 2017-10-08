from dolfin import Mesh, cells, vertices
import numpy as np
import operator


def macro_dofmap(size, space, mesh):
    '''
    For each dof create a set of degrees of freedom which are located
    on macro element of size around the dof.
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
    # dm.cell_dofs() returns all dofs on a cell. Learn which
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
        '''Macro element of size 1 are (typically) the 2 cells that 
        are connected to a (cell exterior) dof. Larger sizes are obtained
        by recursing and join the created size 1 elements.
        '''
        connected_cells = tuple(sorted(v2c(dof_to_vertex_map[dof])))
        elm = set([(connected_cells, dof)])
        # Base case: 2 (1 cell supporting the macroelement of dof)
        if size == 1: return elm

        # Use the 2 boundary dofs of the elm to grow
        dofs = reduce(operator.or_,  
                      (set(dm.cell_dofs(cell)[exterior_idx])-set([dof])
                       for cell in connected_cells))
        # Union of size 1 macro elements (on the leaf level)
        return elm | reduce(operator.or_,
                            (macro_element(d, size-1) for d in dofs))
        
    # Closure for computenig interior dofs of macro elements
    def interior_dofs(elms):
        '''Interior degree od freedom for a macro element'''
        dofs = set()
        for elm in elms:
            all_dofs, exterior_dofs = [], []
            # Fill relative to cell
            cells, dof = elm
            for cell in cells:
                cell_dofs = dm.cell_dofs(cell)
                exterior_cell_dofs = cell_dofs[exterior_idx]
                all_dofs.extend(cell_dofs)
                exterior_dofs.extend(exterior_cell_dofs)
            # Interior dofs of macro element of size 1 are dofs -
            # exterior dofs of cell + dof
            dofs.update(set(all_dofs) - (set(exterior_dofs) - set([dof])))
            # A union of these make up dofs of larger macro element
        return dofs

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
