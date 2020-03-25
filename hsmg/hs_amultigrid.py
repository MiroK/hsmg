import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from dolfin import between, Timer
from hsmg.smoothers import HSAS
# Return instance of Trygve's H^s multigrid. Its __call__ is an
# action of numpy array

def setup(A, M, R, s, bdry_dofs, macro_dofmap, mg_params):
    '''
    Factory function for creating instances of MG methods for fractional Sobolev norms.
    INPUT:
        A: Stiffness matrix on finest level
        M: Mass matrix on finest level
        R: Restriction matrices
        s: Fractionality exponent
        bdry_dofs: Boundary DOFs on each mesh level in hierarchy.
        macro_dofmap: For each level tabulates DOFs for each subdomain, to be used in smoother.
        neg_mg: ("prepost", "bpl") Approach for negative s. 
            "prepost": Pre- and post apply HsMG with 0.5*(1+s), and using stiffness matrix.
            "bpl": MG operator inspired by work of Bramble, Pasciak, and Leyk.
        kwargs: Other parameters
    OUTPUT:
        res: Returns instance of MG-method that works on vectors, and can be called via __call__(b).
    '''
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
            # Set patch-to-DOFs maps for each level:
            if "macro_size" not in self.mg_params.keys():
                self.macro_size = 1
            else:
                assert self.macro_size >= 1

            #self.macro_dms = macro_dofmap(self.macro_size)
            #assert len(self.macro_dms) == self.J
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
                #dofmap_patches = self.macro_dms[k]
                mask = self.masks[k]
                dofmap_patches = [np.array([i]) for i in range(Ak.shape[0])]
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

    # Return different types of preconditioner depending on s:
    assert s > 0
    timer = Timer('setupHsAMG')
    obj = FracLapAMG(A, M, R, s, bdry_dofs, macro_dofmap, mg_params)
    print('HsAMG setup took %g s' % timer.stop())
    
    return obj
