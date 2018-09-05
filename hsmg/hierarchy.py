from dolfin import MeshFunction, refine, Mesh, MeshEditor


def by_refining(seed, nlevels):
    '''Create a mesh hierarchy by uniform refinement of seed.'''

    hierarchy = [seed]
    level = 1
    while level < nlevels:
        mesh = hierarchy[-1]
        where = MeshFunction('bool', mesh, mesh.topology().dim(), True)
        hierarchy.append(refine(mesh, where))

        level += 1
    # Convention to have to finest as first
    return hierarchy[::-1]


def by_coarsening(seed, nlevels):
    '''Create a mesh hierarchy by coarsening the seed.'''
    hierarchy = [seed]
    level = 1
    while level < nlevels:
        mesh = hierarchy[-1]
        hierarchy.append(coarsen(mesh))

        level += 1
    return hierarchy


def coarsen(mesh):
    '''Coarsen of line mesh'''
    # Note that if the mesh is embedded the coarsening as it is implemented
    # here could change the shape, e.g |_ becomes \. The algorithm
    # below works okay for straight segments. In case the mesh is not
    # straight break it to segments, coarsen and merge; fenics_ii-dev
    # has merge path meshes.
    # FIXME: handle other than segments
    
    # This could be done by dolfin.MeshHierarchy.coarsen but that thing
    # segfaults.
    assert mesh.topology().dim() == 1
    ncells = mesh.num_cells()
    assert ncells > 0 and (ncells & (ncells - 1)) == 0

    if ncells == 1: return mesh
    
    # We want a mesh that is a similar to interval or a circle
    mesh.init(0, 1)
    v2c = mesh.topology()(0, 1)
    c2v = mesh.topology()(1, 0)

    endpoints = []
    for v in xrange(mesh.num_vertices()):
        nconnected = len(v2c(v))
        if nconnected == 1:
            endpoints.append(v)
        else:
            assert nconnected == 2

    # Build a path which traverses the mesh
    assert endpoints
    start, end = endpoints

    vertex_idx = [start]  # In the fine mesh
        
    edge = c2v(start)[0]
    while True:
        # s -edge- l -- n --
        v0, v1 = c2v(edge)
        link = v0 if v1 == start else v1
        # Move edge
        e0, e1 = v2c(link)  # Safe by assumptions of 2^k
        edge = e0 if edge == e1 else e1

        # Next start
        v0, v1 = c2v(edge)
        start = v0 if v1 == link else v1
        
        vertex_idx.append(start)
        
        if start == end: break
        # Move edge
        e0, e1 = v2c(start)  # Safe by assumptions of 2^k
        edge = e0 if edge == e1 else e1

    # Coarse mesh
    mesh_vertices = mesh.coordinates()[vertex_idx]
    nvertices, gdim = mesh_vertices.shape
    mesh_cells = [(i, i+1) for i in xrange(nvertices-1)]

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 1, gdim)
    editor.init_vertices(nvertices)
    editor.init_cells(len(mesh_cells))

    # Add vertices
    for index, v in enumerate(mesh_vertices): editor.add_vertex(index, v)

    # Add cells
    for index, c in enumerate(mesh_cells): editor.add_cell(index, *c)

    editor.close()

    return mesh

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import UnitIntervalMesh, cells, near
    from dolfin import BoundaryMesh, UnitSquareMesh
    from dolfin import CompiledSubDomain, SubMesh, UnitCubeMesh
    from numpy.linalg import norm
    
    nlevels = 7
    gdim = 3

    # Test on different meshes:
    if gdim == 1:
        mesh = UnitIntervalMesh(4)
    # Painful 1d in 2d        
    elif gdim == 2:
        mesh = BoundaryMesh(UnitSquareMesh(8, 8), 'exterior')
        subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        CompiledSubDomain('near(x[1], 0)').mark(subdomains, 1)
        mesh = SubMesh(mesh, subdomains, 1)
    # 1d in 3d
    else:
        mesh = BoundaryMesh(UnitCubeMesh(8, 8, 8), 'exterior')
        subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        CompiledSubDomain('near(x[2], 0)').mark(subdomains, 1)
        mesh = SubMesh(mesh, subdomains, 1)

        mesh = BoundaryMesh(mesh, 'exterior')
        subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        CompiledSubDomain('near(x[1], 0)').mark(subdomains, 1)
        mesh = SubMesh(mesh, subdomains, 1)


    gdim = mesh.geometry().dim()
    
    from_coarse = by_refining(mesh, nlevels)
    assert len(from_coarse) == nlevels

    from_fine = by_coarsening(from_coarse[0], nlevels)

    for hierarchy in (from_coarse, from_fine):
        assert all(hierarchy[i].hmin() < hierarchy[i+1].hmin()
                   for i in range(0, len(hierarchy)-1))

    for meshc, meshf in zip(from_coarse, from_fine):
        assert meshc.num_cells() == meshf.num_cells()

        # Correspondence between cells
        for ccell in cells(meshc):
            cmp = ccell.midpoint()

            found = False
            for fcell in cells(meshf):
                fmp = fcell.midpoint()
                found = near(cmp.distance(fmp), 0)
                if found: break
            assert found
            x, y = ccell.get_vertex_coordinates(), fcell.get_vertex_coordinates()
            x = x.reshape((-1, gdim))
            y = y.reshape((-1, gdim))
            # Can be permuted
            assert near(norm(x-y), 0) or near(norm(x-y[::-1]), 0), (x, y)
    print gdim, 'OKAY'
