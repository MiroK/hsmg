# Here we solve the Babuska problem
#   
#   curl(curl(u)) + u = f  in \Omega
#                 u.t = g  in \Gamma
#
# Enforcing bcs weakly leads to saddle point formulation with Lagrange
# multiplier in H^0.5 requiring Schur complement preconditioner based
# on -\Delta ^ 0.5

from fenics_ii.trace_tools.trace_assembler import trace_assemble
from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh

from block import block_mat, block_vec, block_bc
from block.algebraic.petsc import LU, InvDiag

from hsmg import HsNormMG, HsNorm, BP_H1Norm

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
        f_rhs, g_rhs = Constant((1, )*gdim), Expression('sin(pi*(x[0] + x[1]))', degree=1)
    else:
        f_rhs, g_rhs = rhs_data

    S = FunctionSpace(omega_mesh, 'Nedelec 1st kind H(curl)', 1)    
    Q = FunctionSpace(gamma_mesh.mesh, 'DG', 0)
    W = [S, Q]
    
    sigma, p = map(TrialFunction, W)
    tau, q = map(TestFunction, W)

    dxGamma = dx(domain=gamma_mesh.mesh)        
    n_gamma = gamma_mesh.normal('+')          # Outer
    
    a00 = inner(curl(sigma), curl(tau))*dx + inner(sigma, tau)*dx
    a01 = inner(dot(tau('+'), n_gamma), p)*dxGamma
    a10 = inner(dot(sigma('+'), n_gamma), q)*dxGamma

    L0 = inner(f_rhs, tau)*dx
    L1 = inner(g_rhs, q)*dxGamma

    A00 = assemble(a00)
    A01 = trace_assemble(a01, gamma_mesh)
    A10 = trace_assemble(a10, gamma_mesh)

    AA = block_mat([[A00, A01],
                    [A10,   0]])

    bb = block_vec(map(assemble, (L0, L1)))

    P00 = LU(A00)

    bdry = None
    mg_params = {'nlevels': len(hierarchy)}
    mg_params.update(mg_params)

    if precond == 'mg':
        P11 = HsNormMG(Q, bdry, 0.5, mg_params, mesh_hierarchy=hierarchy)
    elif precond == 'eig':
        # Bonito
        P11 = HsNorm(Q, s=0.5)**-1
    else:
        bp_params = {'k': lambda s, N, h: 5.0*1./ln(N),
                     'solver': 'cholesky'}
        P11 = BP_H1Norm(Q, 0.5, bp_params)
        
    # The preconditioner
    BB = block_mat([[P00, 0], [0, P11]])

    return AA, bb, BB, W


def setup_case_2d(**kwargs):
    from mms_setups import curl_curl_2d
    return curl_curl_2d()


def setup_case_3d(**kwargs):
    raise NotImplementedError()


def setup_error_monitor(up, memory):
    from error_convergence import monitor_error, Hcurl_norm, Hs_norm
    return monitor_error(up, [Hcurl_norm, Hs_norm(0.5)], memory)


def setup_fractionality(): return 0.5
