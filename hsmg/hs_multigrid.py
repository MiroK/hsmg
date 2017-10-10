#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from dolfin import between
# Return instance of Trygve's H^s multigrid. Its __call__ is an
# action of numpy array


def setup(A, M, R, bdry_dofs, macro_dofmap):
    '''
    Factory function for creating instances of MG methods for fractional Sobolev norms.
    INPUT:
        A: Stiffness matrix on finest level
        M: Mass matrix on finest level
        R: Restriction matrices
        bdry_dofs: Boundary DOFs on each mesh level in hierarchy.
        macro_dofmap: For each level tabulates DOFs for each subdomain, to be used in smoother.
    OUTPUT:
        res: Returns instance of MG-method that works on vectors, and can be called via __call__(b).
    '''
    # Dummy for testing
    class FracLapMG(object):
        '''Class for MG method''' 
        def __init__(self, A, M, R, macro_dofmap, size=1, bdry_dofs=None, **kwargs):
        #def __init__(self, A, M, R, bdry_dofs, macro_dofmap, ns=1, s=1.0, size=1, eta=1.0):
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
            self.kwargs = kwargs
            if "s" not in self.kwargs.keys():
                self.s = 1.
            else:
                assert between(s, (0.,1.))
            
            if "ns" not in self.kwargs.keys():
                self.ns = 1
            else:
                assert isinstance(self.ns, int)
                assert self.ns >= 1

            if "eta" not in self.kwargs.keys():
                self.eta = 1.
            else:
                assert eta > 0.
            # Set patch-to-DOFs maps for each level:
            assert size >= 1
            self.macro_dms = macro_dofmap(size)
            assert len(self.macro_dms) == self.J
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
            if bdry_dofs is not None:
                self.masks = []
                for i in range(self.J):
                    mask = np.ones( self.As[i].shape[0], dtype=bool)
                    mask[bdry_dofs[i]] = False
                    self.masks.append(mask)
                assert len(self.masks) == self.J
            else:
                self.masks = [[]]*self.J
            # Set smoothers:
            self.set_smoothers()
            # Set coarsest level inverse
            self.set_coarsest_inverse()

        def __getattr__(self, key):
            return self.kwargs[key]
        
        def __setattr__(self, key, value):
            super(FracLapMG, self).__setattr__(key,value)

        def set_smoothers(self):
            '''Method for setting Additive Schwarz smoothers.'''
            self.smoothers = []
            for k in xrange(self.J):
                Ak = self.As[k]
                Mk = self.Ms[k]
                dofmap_patches = self.macro_dms[k]
                mask = self.masks[k]
                
                self.smoothers.append( HSAS(Ak, Mk, self.s, dofmap_patches, mask, eta=self.eta) )

            # Checks:
            assert len(self.smoothers) == self.J

        def multigrid_level(self,j,bj):
            '''Multigrid method on general level.
            INPUT:
                j: integer; level in mesh hierarchy
                bj: vector to apply MG method to.
            RETURN:
                res: Output vector from method.
            '''
            if (j >= self.J-1):
                return self.coarse_solve(bj)
            else:
                S = self.smoothers[j]
                A = self.As[j]
                Rm = self.R[j]
                x = np.zeros(len(bj))
                # Pre-smoothings:
                for i in range(self.ns):
                    #FIXME: Replace residual operator
                    x +=  S(bj - A.dot(x))
                # restrict residual:
                rc = Rm.dot(bj - A.dot(x))
                # coarse grid correction:
                x += Rm.T.dot( self.multigrid_level(j+1,rc) )
                # Post-smoothings:
                for i in range(self.ns):
                    x += S(bj - A.dot(x))
                return x

        def set_coarsest_inverse(self):
            '''Sets the inverse matrix on the coarsest level, once and for all.'''
            mask = self.masks[-1]
            A = self.As[-1].todense()[np.ix_(mask,mask)]
            M = self.Ms[-1].todense()[np.ix_(mask,mask)]
            lam, U = la.eigh( A, b=M, type=1)
            U = np.asarray(U)
            self.Hsinv_coarse = np.dot(lam**(-self.s) * U, U.T)

        def coarse_solve(self, bj):
            '''Solve exactly on coarsest level of mesh hierarchy.'''
            res = bj.copy()
            mask = self.masks[-1]
            res[mask] = np.dot(self.Hsinv_coarse, bj[mask])
            return res

        def __call__(self, b):
            '''Call method.'''
            #return self.As[0].dot(b)
            return self.multigrid_level(0,b)

    return FracLapMG(A,M,R, macro_dofmap, size=1, bdry_dofs=bdry_dofs)


class HSAS(object):
    '''Class for Additive Schwarz operator approximating the Hs norm.'''
    def __init__(self, A, M, s, dms, mask=[], eta=1.0):
        '''Constructor.
        INPUT:
            A: stiffness matrix
            M: mass matrix
            s: (0,1), fractionality
            dms: DOFs on each local patch
            mask: Ignored indices when using operator
            eta: scaling coefficient
        '''
        self.mask = mask
        self.eta = eta
        # Go through each patch:
        print("dms: {}".format(dms))
        B = sp.lil_matrix( A.shape, dtype=float )
        for dofs in dms:
            # Local matrices:
            Aloc = A[dofs][:,dofs].todense()
            Mloc = M[dofs][:,dofs].todense()
            # solve local eigenvalue problem:
            lam, Uloc = la.eigh(Aloc, b=Mloc, type=1)
            Uloc = np.asarray( Uloc )
            # Insert appropriately:
            B[np.ix_(dofs,dofs)] += np.dot(lam**(-s) * Uloc, Uloc.T)
        # Set matrix:
        self.B = sp.csr_matrix(B)

    def __call__(self, b):
        res = np.zeros(self.B.shape[1])
        mask = self.mask
        res[mask] = self.eta * self.B[np.ix_(mask,mask)].dot(b[mask])
        return res
