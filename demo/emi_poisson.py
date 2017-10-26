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

from hsmg import Hs0NormMG

from dolfin import *


def main(markers, subdomains, beta=1E-10):
    '''Solver'''
    kappa_e = Constant(1)
    kappa_i = Constant(1)

    g = Expression('sin(k*pi*(x[0]+x[1]))', k=3, degree=3)

    omega = markers.mesh()
    dim = omega.geometry().dim()
    gamma = EmbeddedMesh(omega, markers, 1, normal=[0.5]*dim)

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
    B22 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=0.5, as_type=PETScMatrix)

    # Alternative B22 block:
    mg_params = {'macro_size': 1,
                 'nlevels': 4,
                 'eta': 0.4}
    bdry = DomainBoundary()
    #B22alt = Hs0NormMG(Q, bdry, 0.5, mg_params)  

    BB = block_mat([[B00, 0, 0],
                    [0, B11, 0],
                    [0, 0, B22]])

    x = AA.create_vec()
    x.randomize()
    AAinv = MinRes(AA, precond=BB, initial_guess=x, tolerance=1e-10, maxiter=500, show=2)

    # Compute solution
    x = AAinv * bb

    niters = len(AAinv.residuals) - 1
    size = V.dim() + Q.dim()
    
    return size, niters
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=int, help='Solve 2d or 3d problem',
                         default=2)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    args = parser.parse_args()

    dim = args.D
    Mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]

    # Interface between interior and exteriorn domains
    gamma = {2: 'near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)',
             3: 'near(std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))), 0.25)'}
    
    gamma = CompiledSubDomain(gamma[dim])

    # Marking interior domains
    interior = {2: 'std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25 ? 1: 0',
                3: 'std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))) < 0.25 ? 1: 0'}
    interior = CompiledSubDomain(interior[dim])
    
    history = []
    for n in [2**i for i in range(2, 1+args.n)]:
        mesh = Mesh(*(n, )*dim)
        markers = FacetFunction('size_t', mesh, 0)
        gamma.mark(markers, 1)
        assert sum(1 for _ in SubsetIterator(markers, 1)) > 0

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
        
        size, niters = main(markers, subdomains)

        msg = 'Problem size %d, current iters %d, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (size, niters, history))
        history.append(niters)
