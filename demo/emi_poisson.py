#  =================
#  |               |
#  |     [ i ]     |
#  |            e  |
#  =================
    
# We solve: Find sigma \in H(div, Omega), tau \in L2(Omega), 
# p in H0.5(Gamma) such that 

# (kappa^-1 sigma, tau) - (u, div tau) + (p, tau . n) = 0
# -(div sigma, v)       + (u, v)                      = 0
# (sigma . n, q)        - beta*(p, q)                 = (g, q)
#
# beta term is due to the system being part of timestepping.
# NOTE: since FEniCS does not have Hdiv multigrid the H(div) preconditioner
# is taken as LU and setting up will take some time.


from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm

from block import block_mat, block_assemble
from block.iterative import MinRes
from block.algebraic.petsc import LU, InvDiag, AMG

from hsmg import HsNormMG
from hsmg.hsquad import BP_H1Norm

from dolfin import *

# Interface between interior and exteriorn domains
gamma = {2: 'near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)',
         3: 'near(std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))), 0.25)'}
    
# Marking interior domains
interior = {2: 'std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25 ? 1: 0',
            3: 'std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))) < 0.25 ? 1: 0'}


def compute_hierarchy(mesh_init, bdry, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0

    if nlevels == 1:
        mesh = mesh_init(*(n, )*dim)

        markers = FacetFunction('size_t', mesh, 0)
        bdry.mark(markers, 1)
        assert sum(1 for _ in SubsetIterator(markers, 1)) > 0
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [EmbeddedMesh(mesh, markers, 1, normal=[0.5]*dim)]

    return compute_hierarchy(mesh_init, bdry, n, 1) + \
        compute_hierarchy(mesh_init, bdry, n/2, nlevels-1)


def setup_system(precond, hierarchy, subdomains, beta=1E-10):
    '''Solver'''
    kappa_e = Constant(1)
    kappa_i = Constant(1)

    g = Expression('sin(k*pi*(x[0]+x[1]))', k=3, degree=3)

    omega = subdomains.mesh()
    gamma = hierarchy[0]  # EmebeddedMesh
    # Hiereachy as Mesh instances
    hierarchy = [h.mesh for h in hierarchy]
    
    S = FunctionSpace(omega, 'RT', 1)        # sigma
    V = FunctionSpace(omega, 'DG', 0)        # u
    Q = FunctionSpace(gamma.mesh, 'DG', 0)   # p
    W = [S, V, Q]

    sigma, u, p = map(TrialFunction, W)
    tau, v, q = map(TestFunction, W)

    dX = Measure('dx', domain=omega, subdomain_data=subdomains)
    dxGamma = dx(domain=gamma.mesh)
        
    n_gamma = gamma.normal('+')          # Outer of inner square
    
    # System - for symmetry
    a00 = inner(Constant(1./kappa_e)*sigma, tau)*dX(0) +\
          inner(Constant(1./kappa_i)*sigma, tau)*dX(1)
    a01 = -inner(u, div(tau))*dX
    a02 = inner(dot(tau('+'), n_gamma), p)*dxGamma

    a10 = -inner(div(sigma), v)*dX
    a11 = inner(u, v)*dX

    a20 = inner(dot(sigma('+'), n_gamma), q)*dxGamma
    a22 = -Constant(beta)*inner(p, q)*dxGamma   

    A00, A01, A10, A11, A22 = map(assemble, (a00, a01, a10, a11, a22))
    A02 = trace_assemble(a02, gamma)
    A20 = trace_assemble(a20, gamma)

    AA = block_mat([[A00, A01, A02],
                    [A10, A11,   0],
                    [A20, 0,   A22]])

    bb = block_assemble([inner(Constant((0, )*dim), tau)*dx,
                         inner(Constant(0), v)*dx,
                         inner(g, q)*dxGamma])

    # Block of Riesz preconditioner
    B00 = LU(assemble(inner(sigma, tau)*dX + inner(div(sigma), div(tau))*dX))
    B11 = InvDiag(assemble(inner(u, v)*dX))

    # (Miro) Gamma here is closed loop so H1_L2_Interpolation norm
    # uses eigenalue problem (-Delta + I) u = lambda I u. Also, no
    # boundary conditions are set
    if precond == 'eig':
        B22 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=0.5, as_type=PETScMatrix)
    elif precond ==  'mg':
        # Alternative B22 block:
        mg_params = {'macro_size': 1,
                     'nlevels': len(hierarchy),
                     'eta': 1.0,
                     'size': 1}

        bdry = None
        B22 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
    
        B22 = BP_H1Norm(Q, 0.5, bp_params)

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
    parser.add_argument('-D', type=int, help='Solve 2d or 3d problem',
                         default=2)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    parser.add_argument('-nlevels', type=int, help='Number of levels for multigrid',
                        default=4)
    parser.add_argument('-Q', type=str, help='iters (with MinRes) or cond (using CGN)',
                        default='iters', choices=['iters', 'cond'])
    parser.add_argument('-B', type=str, help='eig preconditioner or MG preconditioner',
                        default='mg', choices=['eig', 'mg', 'bp'])
    parser.add_argument('-log', type=str, help='Path to file for storing results',
                        default='')

    args = parser.parse_args()

    dim = args.D
    Mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]

    main = iter_solve if args.Q == 'iters' else cond_solve

    # Interface between interior and exteriorn domains
    gamma = CompiledSubDomain(gamma[dim])
    # Marking interior domains
    interior = CompiledSubDomain(interior[dim])

    sizes, history = [], []
    for n in [2**i for i in range(5, 5+args.n)]:
        # Embedded
        hierarchy = compute_hierarchy(Mesh, gamma, n, nlevels=4)

        mesh = Mesh(*(n, )*dim)
        # Setup tags of interior domains 
        subdomains = CellFunction('size_t', mesh, 0)
        
        try:
            for cell in cells(mesh):
                subdomains[cell] = interior.inside(cell.midpoint().array(), False)
        # UiO FEniCS 1.6.0 does not have point array
        except AttributeError:
            for cell in cells(mesh):
                mp = cell.midpoint()
                x = np.array([mp[i] for i in range(dim)])
                subdomains[cell] = interior.inside(x, False)
        
        assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

        system = setup_system(args.B, hierarchy, subdomains)

        size, value = main(system)

        msg = 'Problem size %d, current %s is %g, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (sum(size), args.Q, value, history[::-1]))
        history.append((value, ))
        sizes.append(size)
    # S, V, Q and cond or iter
    args.log and log_results(args, sizes, {0.5: history}, fmt='%d %d %d %g')
