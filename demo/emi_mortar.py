from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm

from block import block_mat, block_vec
from block.algebraic.petsc import AMG

from hsmg import HsNormMG
from hsmg.hsquad import BP_H1Norm

from dolfin import *


# Interface between interior and exteriorn domains
gamma = {2: 'near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)',
         3: 'near(std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))), 0.25)'}
    
# Marking interior domains
interior = {2: 'std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25 ? 1: 0',
            3: 'std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))) < 0.25 ? 1: 0'}


def compute_hierarchy(dim, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0

    if nlevels == 1:
        if dim == 2:
            mesh = RectangleMesh(Point(0.25, 0.25), Point(0.75, 0.75), n/2, n/2)
        else:
            mesh = BoxMesh(Point(0.25, 0.25, 0.25), Point(0.75, 0.75, 0.75), n/2, n/2, n/2)
        bdry = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
        DomainBoundary().mark(bdry, 1)

        return [EmbeddedMesh(mesh, bdry, 1, normal=[0.5]*dim)]

    return compute_hierarchy(dim, n, 1) + compute_hierarchy(dim, n/2, nlevels-1)


def setup_system((mesh0, mesh1), hierarchy, precond, mg_params_):
    '''Solver'''
    gamma = hierarchy[0]
    
    Ve = FunctionSpace(mesh0, 'CG', 1)
    Vi = FunctionSpace(mesh1, 'CG', 1)
    Q = FunctionSpace(gamma.mesh, 'CG', 1)
    W = [Ve, Vi, Q]
    
    ue, ui, p = map(TrialFunction, W)
    ve, vi, q = map(TestFunction, W)

    kappa_e = Constant(1.5)
    kappa_i = Constant(1)
    # Blocks of lhs
    ae =  inner(kappa_e*grad(ue), grad(ve))*dx + inner(ue, ve)*dx
    Ae = assemble(ae)

    ai =  inner(kappa_i*grad(ui), grad(vi))*dx + inner(ui, vi)*dx
    Ai = assemble(ai)
    
    dxGamma = Measure('dx', domain=gamma.mesh)

    be = inner(ue, q)*dxGamma
    Be = trace_assemble(be, gamma)
    
    bi = inner(ui, q)*dxGamma
    Bi = trace_assemble(bi, gamma)
    Bi *= -1

    # Transp
    beT = inner(ve, p)*dxGamma
    BeT = trace_assemble(beT, gamma)
    
    biT = inner(vi, p)*dxGamma
    BiT = trace_assemble(biT, gamma)
    BiT *= -1

    k = Constant(1)
    f = sin(k*pi*sum(SpatialCoordinate(gamma.mesh)))
    # Block of rhs
    be = assemble(inner(Constant(0), ve)*dx)
    bi = assemble(inner(Constant(0), vi)*dx)
    bQ = assemble(inner(f, q)*dx)

    # The linear system
    AA = block_mat([[Ae, 0, BeT],
                    [0, Ai, BiT],
                    [Be, Bi,  0]])

    bb = block_vec([be, bi, bQ])
    
    # Preconditioner using broken H1 norm
    B00 = AMG(assemble(inner(grad(ue), grad(ve))*dx + inner(ue, ve)*dx))
    B11 = AMG(assemble(inner(grad(ui), grad(vi))*dx + inner(ui, vi)*dx))
    
    if precond == 'eig':
        B22 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=-0.5, as_type=PETScMatrix)
    elif precond ==  'mg':
        # Hiereachy as Mesh instances
        hierarchy = [h.mesh for h in hierarchy]          
        # Alternative B22 block:
        mg_params = {'nlevels': len(hierarchy)}
        mg_params.update(mg_params_)

        bdry = None
        B22 = HsNormMG(Q, bdry, -0.5, mg_params, mesh_hierarchy=hierarchy)
    # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
    
        B22 = BP_H1Norm(Q, -0.5, bp_params)

    BB = block_mat([[B00, 0, 0],
                    [0, B11, 0],
                    [0, 0, B22]])

    return AA, bb, BB
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import numpy as np
    from common import log_results, iter_solve, cond_solve

    
    parser = argparse.ArgumentParser()
    # What
    parser.add_argument('-D', type=int, help='Solve 2d or 3d problem',
                         default=2)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-Q', type=str, help='iters (with MinRes) or cond (using CGN)',
                        default='iters', choices=['iters', 'cond'])
    # How
    parser.add_argument('-B', type=str, help='eig preconditioner or MG preconditioner',
                        default='mg', choices=['eig', 'mg', 'bp'])
    # Store
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                        default='')
    # Iterative settings
    parser.add_argument('-tol', type=float, help='Relative tol for Krylov',
                        default=1E-12)
    parser.add_argument('-eta', type=float, help='eta parameter for MG smoother',
                         default=1.0)
    parser.add_argument('-mes', type=int, help='Macro element size for MG smoother',
                        default=1)
    parser.add_argument('-nlevels', type=int, help='Number of levels for multigrid',
                        default=4)

    args = parser.parse_args()

    dim = args.D
    Mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]

    main = iter_solve if args.Q == 'iters' else cond_solve

    # Interface between interior and exteriorn domains
    gamma = CompiledSubDomain(gamma[dim])
    # Marking subdomains
    subdomains = CompiledSubDomain(interior[dim])

    sizes, history = [], []
    for level, n in enumerate([2**i for i in range(4, 4+args.n)], 1):
        # Setup the interior/exterior domains
        mesh = Mesh(*(n, )*dim)
        cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

        try:
            for cell in cells(mesh):
                cell_f[cell] = subdomains.inside(cell.midpoint().array(), False)
        # UiO FEniCS 1.6.0 does not have point array
        except AttributeError:
            for cell in cells(mesh):
                mp = cell.midpoint()
                x = np.array([mp[i] for i in range(dim)])
                cell_f[cell] = subdomains.inside(x, False)

        mesh0 = SubMesh(mesh, cell_f, 0)
        mesh1 = SubMesh(mesh, cell_f, 1)

        # The interface domain and the hierarchy on top of it
        hierarchy = compute_hierarchy(dim, n, nlevels=args.nlevels)

        system = setup_system((mesh0, mesh1), hierarchy,
                              args.B,
                              mg_params_={'macro_size': args.mes, 'eta': args.eta})

        size, value = main(system, args.tol)

        msg = '(%d) Problem size %d[%r], current %s is %g, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (level, sum(size), size, args.Q, value, history[::-1]))
        history.append((value, ))
        sizes.append(size)
        
        # spaces and cond or iter
        args.log and log_results(args, sizes, {-0.5: history}, fmt='%d %d %d %.16f')
