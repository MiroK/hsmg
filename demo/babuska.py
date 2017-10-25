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
from block.iterative import MinRes
from block.algebraic.petsc import AMG

from dolfin import *


def main(mesh):
    '''Solver'''
    omega_mesh = mesh
    facet_f = FacetFunction('size_t', omega_mesh, 0)
    CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)

    gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

    # Space of u and the Lagrange multiplier
    V = FunctionSpace(omega_mesh, 'CG', 1)
    Q = FunctionSpace(gamma_mesh.mesh, 'CG', 1)

    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)

    dxGamma = Measure('dx', domain=gamma_mesh.mesh)

    a00 = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    a01 = inner(p, v)*dxGamma
    a10 = inner(u, q)*dxGamma
    
    L0 = inner(Constant(1), v)*dx
    L1 = inner(Expression('sin(pi*(x[0] + x[1]))', degree=1), q)*dxGamma

    # Blocks
    A00 = assemble(a00)
    A01 = trace_assemble(a01, gamma_mesh)
    A10 = trace_assemble(a10, gamma_mesh)

    b0 = assemble(L0)
    b1 = assemble(L1)
    
    # System
    AA = block_mat([[A00, A01], [A10, 0]])
    bb = block_vec([b0, b1])
    
    # Preconditioner blocks
    P00 = AMG(A00)
    # Trace of H^1 is H^{1/2} and the dual is H^{-1/2}
    # FIXME: this should be replaced with multigrid
    P11 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=-0.5, as_type=PETScMatrix)
    # The preconditioner
    BB = block_mat([[P00, 0], [0, P11]])

    x = AA.create_vec()
    x.randomize()

    AAinv = MinRes(AA, precond=BB, tolerance=1e-10, maxiter=500, show=2)

    # Compute solution
    x = AAinv * bb

    niters = len(AAinv.residuals) - 1
    size = V.dim() + Q.dim()
    
    return size, niters

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=int, help='Solve 2d or 3d problem',
                         default=2)
    parser.add_argument('-n', type=int, help='Number of refinements of initial mesh',
                        default=4)
    args = parser.parse_args()

    dim = args.D
    Mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim]

    history = []
    for n in [2**i for i in range(2, 2+args.n)]:
        mesh = Mesh(*(n, )*dim)
        size, niters = main(mesh)

        msg = 'Problem size %d, current iters %d, previous %r'
        print '\033[1;37;31m%s\033[0m' % (msg % (size, niters, history))
        history.append(niters)
