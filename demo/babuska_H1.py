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
        # CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
        DomainBoundary().mark(facet_f, 1)
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


def setup_case_2d():
    from mms_setups import babuska_H1_2d
    return babuska_H1_2d()


def setup_case_3d():
    from mms_setups import babuska_H1_3d
    return babuska_H1_3d()


def setup_error_monitor(true, memory):
    from error_convergence import monitor_error, H1_norm, Hs_norm
    return monitor_error(true, [H1_norm, Hs_norm(-0.5)], memory)


def setup_fractionality(): return -0.5
