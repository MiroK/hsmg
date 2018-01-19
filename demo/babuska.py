# Here we solve the Babuska problem
#   
#   -\Delta u + u = f  in \Omega
#              Tu = g  in \Gamma
#
# Enforcing bcs weakly leads to saddle point formulation with Lagrange
# multiplier in H^-0.5 requiring Schur complement preconditioner based
# on -\Delta ^ -0.5

from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from block import block_mat, block_vec, block_bc
from block.algebraic.petsc import AMG

from hsmg import HsNormMG
from hsmg.hsquad import BP_H1Norm

from dolfin import *
import numpy as np


def setup_system(rhs_data, precond, meshes, mg_params_):
    '''Solver'''
    omega_mesh = meshes[0]
    # Extract botttom edge meshes
    hierarchy = []
    gamma_mesh = None
    for mesh in meshes:
        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
        CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        # DomainBoundary().mark(facet_f, 1)
        gmesh = EmbeddedMesh(mesh, facet_f, 1)        

        if gamma_mesh is None: gamma_mesh = gmesh

        hierarchy.append(gmesh.mesh)
        
    # Space of u and the Lagrange multiplier
    V = FunctionSpace(omega_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'CG', 1)

    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)

    dxGamma = Measure('dx', domain=gamma_mesh.mesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(p, v)*dxGamma
    a10 = inner(u, q)*dxGamma

    if rhs_data is None:
        f, g = Constant(1), Expression('sin(pi*(x[0] + x[1]))', degree=1)
    else:
        f, g = rhs_data
        
    L0 = inner(f, v)*dx
    L1 = inner(g, q)*dxGamma

    # Blocks
    A00 = assemble(a00)
    A01 = trace_assemble(a01, gamma_mesh)
    A10 = trace_assemble(a10, gamma_mesh)

    b0 = assemble(L0)
    b1 = assemble(L1)
    
    # System
    AA = block_mat([[A00, A01], [A10, 0]])
    bb = block_vec([b0, b1])

    print 'Assembled AA'
    # Preconditioner blocks
    P00 = AMG(A00)

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)
    
    # Trace of H^1 is H^{1/2} and the dual is H^{-1/2}
    if precond == 'mg':
        P11 = HsNormMG(Q, bdry, -0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        P11 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=-0.5, as_type=PETScMatrix)            # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P11 = BP_H1Norm(Q, -0.5, bp_params)
    print 'Setup B'
        
    # The preconditioner
    BB = block_mat([[P00, 0], [0, P11]])

    return AA, bb, BB, [V, Q]

    
def compute_hierarchy(mesh_init, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0

    if nlevels == 1:
        mesh = mesh_init(*(n, )*dim)
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [mesh]

    return compute_hierarchy(mesh_init, n, 1) + compute_hierarchy(mesh_init, n/2, nlevels-1)

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import numpy as np
    from common import log_results, cond_solve, iter_solve, direct_solve

    
    parser = argparse.ArgumentParser()
    # What
    parser.add_argument('-D', type=int, help='Solve 2d or 3d problem',
                         default=2)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-Q', type=str, help='iters (with MinRes) or cond (using CGN)',
                        default='iters', choices=['iters', 'cond', 'sane'])
    # How
    parser.add_argument('-B', type=str, help='eig preconditioner or MG preconditioner',
                        default='MG', choices=['eig', 'mg', 'bp'])
    
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                        default='')
    # Iter settings
    parser.add_argument('-tol', type=float, help='Relative tol for Krylov',
                        default=1E-12)
    parser.add_argument('-nlevels', type=int, help='Number of levels for multigrid',
                        default=4)
    parser.add_argument('-eta', type=float, help='eta parameter for MG smoother',
                        default=1.0)
    parser.add_argument('-mes', type=int, help='Macro element size for MG smoother',
                        default=1)
    # Keep an eye on the error of the converged solution
    parser.add_argument('-error', type=int, help='Compare to analytical solution',
                        default=0)

    args = parser.parse_args()

    dim = args.D
    Mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]

    main = {'iters': iter_solve,
            'cond': cond_solve,
            'sane': direct_solve}[args.Q]

    # What rhs to use and monitoring
    if args.error:
        from error_convergence import monitor_error, H1_norm, Hs_norm
        from mms_setups import babuska_H1_2d

        up, fg = babuska_H1_2d()
        
        memory = []
        monitor = monitor_error(up, [H1_norm, Hs_norm(0.5)], memory)
    else:
        memory, fg, monitor = None, None, None

        
    sizes, history = [], []
    for n in [2**i for i in range(5, 5+args.n)]:
        # Embedded
        hierarchy = compute_hierarchy(Mesh, n, nlevels=args.nlevels)
        print 'Hierarchies', [mesh.num_vertices() for mesh in hierarchy]
        
        setup = setup_system(fg, args.B, hierarchy, mg_params_={'macro_size': args.mes, 'eta': args.eta})
        size, value, u = main(setup, args.tol)

        if monitor is not None: monitor.send(u)

        msg = 'Problem size %d[%s], current %s is %g, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (sum(size), size, args.Q, value, history[::-1]))
        history.append((value, ))
        sizes.append(size)
        
    # S, V, Q and cond or iter
    args.log and log_results(args, sizes, {-0.5: history}, fmt='%d %d %.16f', cvrg=memory)
