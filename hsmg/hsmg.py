from dolfin import SubDomain, CompiledSubDomain, between, Constant
from dolfin import DirichletBC, inner, grad, dx, assemble_system
from dolfin import FacetFunction, TrialFunction, TestFunction
from dolfin import Vector
from dolfin import CellSize, avg, dot, jump, dS, ds

from block.object_pool import vec_pool
from block.block_base import block_base

from functools import partial

import macro_element
import hs_multigrid
import restriction
import hierarchy
import utils


class HsNormMGBase(block_base):
    '''
    Multi grid approximation of fraction H^s norm. The norm is defined
    in terms of eigenvalue problem: Find u \in V, lambda in \mathbb{R} 
    such that for all v \in V a(u, v) = m(u, v).
    '''
    def __init__(self, a, m, bdry, s, mg_params, mesh_hierarchy=None):
        # The input here is
        # a, m the bilinear forms
        # bdry; an instance of SubDomain class which marks the boundaries
        # s: fraction exponent
        # mg_prams: a dictionary of multigrid parameters
        # ------------------------------------------------------------
        # mesh_hierarchy: this is currently a part of signature because
        # construction of dolfin's MeshHierarchy by coarsening the mesh
        # of V does not work. We have a hand-made and rather limited
        # implementation of this functionality for 1d meshes
        
        # Same function space for a 
        V = set(arg.function_space() for arg in a.arguments())
        assert len(V) == 1
        V = V.pop()
        # and m
        assert V in set(arg.function_space() for arg in m.arguments())
        # Limit to scalar valued functions
        assert V.dolfin_element().value_rank() == 0

        # If coarsening could preserve boundary tags bcs by markers could
        # be added
        assert bdry is None or isinstance(bdry, (SubDomain, CompiledSubDomain))
        # Keep s to where we know it works
        assert between(s, (-1, 1.))

        # 0 is the finest_mesh
        nlevels = mg_params['nlevels']
        if mesh_hierarchy is None:
            mesh_hierarchy = hierarchy.by_coarsening(V.mesh(), nlevels)

        # If el is the finite element of V we build here operators
        # taking FunctionSpace(mesh_hierarchy[i], el) to
        # FunctionSpace(mesh_hierarchy[i+1], el)
        R = restriction.restriction_matrix(V, mesh_hierarchy)

        # The function which given macro element size produces for each
        # level a map dof -> macro dofs
        macro_dofmaps = partial(macro_element.macro_dofmap,
                               space=V,
                               mesh=mesh_hierarchy)
        # e.g. macro_dofmaps(1)  # of size 1

        if bdry is not None:
            # For each level keep track of boundary dofs
            bdry_dofs = restriction.Dirichlet_dofs(V, bdry, mesh_hierarchy)

            # Finally assemble the matrices of finest level
            mesh = mesh_hierarchy[0]
            bdries = FacetFunction('size_t', mesh, 0)
            bdry.mark(bdries, 1)
            bcs_V = DirichletBC(V, Constant(0), bdries, 1)
        else:
            bdry_dofs = [set()*len(mesh_hierarchy)]
            bcs_V = None
                            
        # FIXME: boundary conditions are built into the system, okay?
        L = inner(Constant(0), TestFunction(V))*dx
        A, _ = assemble_system(a, L, bcs_V)
        M, _ = assemble_system(m, L, bcs_V)

        A, M = map(utils.to_csr_matrix, (A, M))
        # FIXME: Setup multigrid here
        self.mg = hs_multigrid.setup(A, M, R, bdry_dofs, macro_dofmaps, mg_params)
        self.size = V.dim()
        
    # Implementation of cbc.block API --------------------------------
    def matvec(self, b):
        # numpy -> numpy
        x_values = self.mg(b.array())  
        # Fill in dolfin Vector
        x = self.create_vec(dim=0)
        x.set_local(x_values); x.apply('insert')
        return x

    @vec_pool
    def create_vec(self, dim=1):
        return Vector(None, self.size)
        
        
class HsNormMG(HsNormMGBase):
    '''
    Multi grid approximation of fraction H^s norm. The norm is defined
    in terms of eigenvalue problem: Find u \in V, lambda in \mathbb{R} 
    such that for all v \in V

        (grad(u), grad(v))*dx + (u, v)*dx = lambda inner(u, v)*dx

    Limit to Lagrange elements
    '''
    def __init__(self, V, bdry, s, mg_params, mesh_hierarchy=None):
        u, v = TrialFunction(V), TestFunction(V)

        if V.ufl_element().family() == 'Lagrange':
            a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
        else:
            assert V.ufl_element().family() == 'Discontinuous Lagrange'
            # For now keep this with only for piecewise constants
            assert V.ufl_element().degree() == 0
            
            h = CellSize(V.mesh())
            h_avg = avg(h)

            a = h_avg**(-1)*dot(jump(v), jump(u))*dS + h**(-1)*dot(u, v)*ds
                
        m = inner(u, v)*dx
        # Note the introduction
        HsNormMGBase.__init__(self, m+a, m, bdry, s, mg_params, mesh_hierarchy)

        
class Hs0NormMG(HsNormMGBase):
    '''
    Multi grid approximation of fraction H_0^s norm. The norm is defined
    in terms of eigenvalue problem: Find u \in V, lambda in \mathbb{R} 
    such that for all v \in V

        (grad(u), grad(v))*dx = lambda inner(u, v)*dx
 
    NOTE: bcs are a must here.
    '''
    def __init__(self, V, bdry, s, mg_params, mesh_hierarchy=None):
        assert bdry is not None

        u, v = TrialFunction(V), TestFunction(V)

        if V.ufl_element().family() == 'Lagrange':
            a = inner(grad(u), grad(v))*dx
        else:
            assert V.ufl_element().family() == 'Discontinuous Lagrange'
            # For now keep this with only for piecewise constants
            assert V.ufl_element().degree() == 0
            
            h = CellSize(V.mesh())
            h_avg = avg(h)

            a = h_avg**(-1)*dot(jump(v), jump(u))*dS
        
        m = inner(u, v)*dx

        HsNormMGBase.__init__(self, a, m, bdry, s, mg_params, mesh_hierarchy)
