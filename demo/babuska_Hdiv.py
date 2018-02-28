# Here we solve the Babuska problem
#   
#   -\Delta u + u = f  in \Omega
#       grad(u).n = g  in \Gamma
#
# A mixed for is used with
#
#     sigma - grad(u) = 0
#     div(sigma) - u = -f
#     sigma.n        = g
#
# So we endup with
#    (sigma, tau) + (u, div(tau))  (p, tau.n) = 0
#    (dig(sigma), v) -(u, v)                  = -(f, v)
#    (sigma.n, q)                             = (g, q)
#
# Enforcing bcs weakly leads to saddle point formulation with Lagrange
# multiplier in H^0.5 requiring Schur complement preconditioner based
# on -\Delta ^ 0.5. Here p = -u on the boundary

from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from block import block_mat, block_vec, block_bc
from block.algebraic.petsc import LU, InvDiag

from hsmg import HsNormMG
from hsmg.hsquad import BP_H1Norm

from dolfin import *
import numpy as np


def setup_system(rhs_data, precond, meshes, mg_params_, sys_params):
    '''Solver'''
    omega_mesh = meshes[0]
    # Extract botttom edge meshes
    hierarchy = []
    gamma_mesh = None
    for mesh in meshes:
        facet_f = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
        gdim = mesh.geometry().dim()
        # CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        DomainBoundary().mark(facet_f, 1)
        gmesh = EmbeddedMesh(mesh, facet_f, 1, [0.5, ]*gdim)        

        if gamma_mesh is None: gamma_mesh = gmesh

        hierarchy.append(gmesh.mesh)
        
    if rhs_data is None:
        f, g = Constant(1), Expression('sin(pi*(x[0] + x[1]))', degree=1)
    else:
        f, g = rhs_data

    S = FunctionSpace(omega_mesh, 'RT', 1)        # sigma
    V = FunctionSpace(omega_mesh, 'DG', 0)        # u
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 0)   # p
    W = [S, V, Q]

    sigma, u, p = map(TrialFunction, W)
    tau, v, q = map(TestFunction, W)

    dxGamma = dx(domain=gamma_mesh.mesh)        
    n_gamma = gamma_mesh.normal('+')          # Outer

    # System - for symmetry
    a00 = inner(sigma, tau)*dx
    a01 = inner(u, div(tau))*dx
    a02 = inner(dot(tau('+'), n_gamma), p)*dxGamma
    
    a10 = inner(div(sigma), v)*dx
    a11 = -inner(u, v)*dx # FIXme: double check sign here
    
    a20 = inner(dot(sigma('+'), n_gamma), q)*dxGamma

    L0 = inner(Constant((0, )*gdim), tau)*dx
    L1 = inner(-f, v)*dx
    L2 = inner(g, q)*dxGamma

    A00, A01, A10, A11 = map(assemble, (a00, a01, a10, a11))
    A02 = trace_assemble(a02, gamma_mesh)
    A20 = trace_assemble(a20, gamma_mesh)

    AA = block_mat([[A00, A01, A02],
                    [A10, A11,   0],
                    [A20, 0,     0]])

    bb = block_vec(map(assemble, (L0, L1, L2)))

    print 'Assembled AA'
    # Preconditioner blocks
    P00 = LU(assemble(inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx))
    P11 = InvDiag(assemble(inner(u, v)*dx))

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params_)
    
    # Trace of Hdiv is H^{-1/2} and the dual is H^{1/2}
    if precond == 'mg':
        P22 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        P22 = H1_L2_InterpolationNorm(Q).get_s_norm_inv(s=0.5, as_type=PETScMatrix)            # Bonito
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P22 = BP_H1Norm(Q, 0.5, bp_params)
    print 'Setup B'
        
    # The preconditioner
    BB = block_mat([[P00, 0, 0], [0, P11, 0], [0, 0, P22]])

    return AA, bb, BB, W


def setup_case_2d(**kwargs):
    from mms_setups import babuska_Hdiv_2d
    return babuska_Hdiv_2d()


def setup_case_3d(**kwargs):
    from mms_setups import babuska_Hdiv_3d
    return babuska_Hdiv_3d()


def setup_error_monitor(up, memory):
    from error_convergence import monitor_error, Hdiv_norm, L2_norm, Hs_norm
    return monitor_error(up, [Hdiv_norm, L2_norm, Hs_norm(0.5)], memory)


def setup_fractionality(): return 0.5
