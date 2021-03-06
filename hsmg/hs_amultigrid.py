import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from dolfin import between, Timer
from hsmg.hs_multigrid import _mass_lump_inv
from hsmg.smoothers import HSAS
# Return instance of Trygve's H^s multigrid. Its __call__ is an
# action of numpy array

def setup(A, M, R, s, bdry_dofs, macro_dofmap, mg_params, neg_mg):
    '''
    Factory function for creating instances of MG methods for fractional Sobolev norms.
    INPUT:
        A: Stiffness matrix on finest level
        M: Mass matrix on finest level
        R: Restriction matrices
        s: Fractionality exponent
        bdry_dofs: Boundary DOFs on each mesh level in hierarchy.
        macro_dofmap: For each level tabulates DOFs for each subdomain, to be used in smoother.
    OUTPUT:
        res: Returns instance of MG-method that works on vectors, and can be called via __call__(b).
    '''
    # Positive s
    class FracLapAMG(object):
        '''Class for MG method''' 
        def __init__(self, A, M, R, s, bdry_dofs, macro_dofmap, mg_params):
            '''Constructor
            INPUT:
                A: Stiffness matrix in finest level.
                M: Mass matrix on finest level
                R: Restriction matrices
                bdry_dofs: Boundary DOFs on each level
                macro_dofmap: Partial mapping to get macro_element DOFs
                size: >=1, size of additive schwarz patches
                ns: Number of smoothings
                s: (0,1), fractionality
                eta: Scaling coefficient for smoother.
            '''
            self.As = [A,]
            self.Ms = [M,]
            self.R = R
            self.J = len(R) + 1
            self.mg_params = mg_params

            assert between(s, (0.,1.))
            self.s = s
            if "eta" not in self.mg_params.keys():
                self.eta = 1.0
            else:
                assert self.eta > 0.

            # Set up matrices:
            for Rm in self.R:
                Af = self.As[-1]
                Mf = self.Ms[-1]

                Ac = Af.dot(Rm.T)
                Ac = Rm.dot(Ac)
                self.As.append(Ac)
                
                Mc = Mf.dot(Rm.T)
                Mc = Rm.dot(Mc)
                self.Ms.append(Mc)
            # Checks
            assert len(self.As) == self.J
            assert len(self.Ms) == self.J
            # Set masks:
            self.masks = []
            if bdry_dofs is not None:
                for i in range(self.J):
                    mask = np.ones( self.As[i].shape[0], dtype=bool)
                    mask[bdry_dofs[i]] = False
                    self.masks.append(mask)
            # (Miro) for no bcs all dofs are active
            else:
                for i in range(self.J):
                    mask = np.ones( self.As[i].shape[0], dtype=bool)
                    self.masks.append(mask)

            assert len(self.masks) == self.J

            self.macro_dms = macro_dofmap
            
            # Set smoothers:
            self.set_smoothers()
            # Set coarsest level inverse
            self.set_coarsest_inverse()

        def __getattr__(self, key):
            return self.mg_params[key]
        
        def __setattr__(self, key, value):
            super(FracLapAMG, self).__setattr__(key,value)

        def set_smoothers(self):
            '''Method for setting Additive Schwarz smoothers.'''
            self.smoothers = []
            for k in xrange(self.J):
                Ak = self.As[k]
                Mk = self.Ms[k]

                dofmap_patches = self.macro_dms[k]
                mask = self.masks[k]
                # FIXME: masks for bcs!
                S = HSAS(Ak, Mk, self.s, dofmap_patches, mask, eta=self.eta)
                self.smoothers.append( S )
            # Checks:
            assert len(self.smoothers) == self.J

        def bpx_level(self, j, bj):
            '''BPX preconditioner on level j.'''
            # Coarsest level:
            if (j >= self.J-1):
                return self.coarse_solve(bj)
            else:
                S = self.smoothers[j]
                Rm = self.R[j]
                # Restrict and apply:
                b_coarse = Rm.dot(bj)

                b_coarse[~self.masks[j+1]] = 0.
                
                x_coarse = self.bpx_level(j+1, b_coarse)
                # Prolong and add:
                x_coarse = Rm.T.dot( x_coarse )
                x_fine = S(bj)
                
                return x_fine + x_coarse

        def set_coarsest_inverse(self):
            '''Sets the inverse matrix on the coarsest level, once and for all.'''
            mask = self.masks[-1]
            A = self.As[-1].todense()[np.ix_(mask,mask)]
            M = self.Ms[-1].todense()[np.ix_(mask,mask)]
            lam, U = la.eigh( A, b=M, type=1)
            U = np.asarray(U)
            self.Hsinv_coarse = np.dot(U * lam**(-self.s), U.T)

        def coarse_solve(self, bj):
            '''Solve exactly on coarsest level of mesh hierarchy.'''
            res = bj.copy()
            mask = self.masks[-1]
            res[mask] = np.dot(self.Hsinv_coarse, bj[mask])
            return res

        def __call__(self, b):
            '''Call method.'''
            return self.bpx_level(0,b)
    # Negative s
    class NegFracLapAMG2(object):
        '''Class for MG method for negative fractionality. Alternative approach.''' 
        def __init__(self, A, M, R, s, bdry_dofs, macro_dofmap, mg_params):
            '''Constructor
            INPUT:
                A: Stiffness matrix in finest level.
                M: Mass matrix on finest level
                R: Restriction matrices
                bdry_dofs: Boundary DOFs on each level
                macro_dofmap: Partial mapping to get macro_element DOFs
                size: >=1, size of additive schwarz patches
                ns: Number of smoothings
                s: (0,1), fractionality
                eta: Scaling coefficient for smoother.
            '''
            self.As = [A,]
            self.Ms = [M,]
            # Mass lumped matrix:
            self.Gs = [_mass_lump_inv(M),]
            self.R = R
            self.J = len(R) + 1
            self.mg_params = mg_params

            assert between(s, (-1.0,0.0))
            self.s = s
            if "eta" not in self.mg_params.keys():
                self.eta = 1.0
            else:
                assert self.eta > 0.

            # Set up matrices:
            for Rm in self.R:
                Af = self.As[-1]
                Mf = self.Ms[-1]

                Ac = Af.dot(Rm.T)
                Ac = Rm.dot(Ac)
                self.As.append(Ac)
                
                Mc = Mf.dot(Rm.T)
                Mc = Rm.dot(Mc)
                self.Ms.append(Mc)
                self.Gs.append(_mass_lump_inv(Mc))
            # Checks
            assert len(self.As) == self.J
            assert len(self.Ms) == self.J
            assert len(self.Gs) == self.J
            # Set masks:
            self.masks = []
            if bdry_dofs is not None:
                for i in range(self.J):
                    mask = np.ones( self.As[i].shape[0], dtype=bool)
                    mask[bdry_dofs[i]] = False
                    self.masks.append(mask)
            # (Miro) for no bcs all dofs are active
            else:
                for i in range(self.J):
                    mask = np.ones( self.As[i].shape[0], dtype=bool)
                    self.masks.append(mask)
                    
            assert len(self.masks) == self.J

            self.macro_dms = macro_dofmap
            # Set smoothers:
            self.set_smoothers()
            # Set coarsest level inverse
            self.set_coarsest_inverse()

        def __getattr__(self, key):
            return self.mg_params[key]
        
        def __setattr__(self, key, value):
            super(NegFracLapAMG2, self).__setattr__(key,value)

        def set_smoothers(self):
            '''Method for setting Additive Schwarz smoothers.'''
            self.smoothers = []
            for k in xrange(self.J):
                Ak = self.As[k]
                Mk = self.Ms[k]
                dofmap_patches = self.macro_dms[k]
                mask = self.masks[k]
                # NOTE: 1+s as input parameter to smoother
                S = HSAS(Ak, Mk, 1+self.s, dofmap_patches, mask, eta=self.eta)
                self.smoothers.append( S )

            # Checks:
            assert len(self.smoothers) == self.J

        def bpx_level(self, j, bj):
            '''BPX preconditioner on level j.'''
            # Coarsest level:
            if (j >= self.J-1):
                return self.coarse_solve(bj)
            else:
                S = self.smoothers[j]
                G = self.Gs[j]
                A = self.As[j]
                Rm = self.R[j]
                # Restrict and apply:
                b_coarse = Rm.dot(bj)

                b_coarse[~self.masks[j+1]] = 0.
                
                x_coarse = self.bpx_level(j+1, b_coarse)
                # Prolong and add:
                x_coarse = Rm.T.dot( x_coarse )
                x_fine = 0.5*G.dot( A.dot( S(bj) ) )
                x_fine += 0.5*S( A.dot( G.dot(bj) ) )
                
                return x_fine + x_coarse

        def set_coarsest_inverse(self):
            '''Sets the inverse matrix on the coarsest level, once and for all.'''
            mask = self.masks[-1]
            A = self.As[-1].todense()[np.ix_(mask,mask)]
            M = self.Ms[-1].todense()[np.ix_(mask,mask)]
            lam, U = la.eigh( A, b=M, type=1)
            U = np.asarray(U)
            self.Hsinv_coarse = np.dot(U * lam**(-self.s), U.T)

        def coarse_solve(self, bj):
            '''Solve exactly on coarsest level of mesh hierarchy.'''
            res = bj.copy()
            mask = self.masks[-1]
            res[mask] = np.dot(self.Hsinv_coarse, bj[mask])
            return res

        def __call__(self, b):
            '''Call method.'''
            return self.bpx_level(0,b)

    class NegFracLapAMG(object):
        ''' BPX preconditioner for fractional laplacian with negative exponent.'''
        def __init__(self, A, M, R, s, bdry_dofs, macro_dofmap, mg_params):
            self.A = A
            assert between( s, (-1.,0.) )
            self.BPX = FracLapAMG(A, M, R, 0.5*(1.+s), bdry_dofs, macro_dofmap, mg_params)

        def __call__(self, b):
            x0 = self.BPX(b)
            b1 = self.A.dot(x0)
            x1 = self.BPX(b1)
            return x1
    #
    # Dispatch on s --------------------------------------------------
    # 
    timer = Timer('setupHsAMG')
    if s > 0:
        mg = FracLapAMG
    else:
        if neg_mg == 'bpl':
            mg = NegFracLapAMG2
        else:
            mg = NegFracLapAMG
    mg = mg(A, M, R, s, bdry_dofs, macro_dofmap, mg_params)
    print('HsAMG setup took %g s' % timer.stop())
    
    return mg
