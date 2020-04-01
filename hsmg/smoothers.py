import numpy as np
import scipy.sparse as sp
import scipy.linalg as la


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

        # We can compote more efficienlty if the smoothers are pointwise
        # rather than patchwise
        if set(map(len, dms)) == set((1, )):
            dms = np.hstack(dms)
            
            Al = A.diagonal()[dms]
            Ml = M.diagonal()[dms]

            Uloc = Ml**-0.5
            lam  = Al*Uloc**2

            # We want the matrix shape as A
            #diag = np.zeros(A.shape[0])
            diag = Uloc**2*lam**(-s)
            
            self.B = sp.csr_matrix(sp.diags(diag))
            return None

        # Go through each patch:
        B = sp.lil_matrix( A.shape, dtype=float )
        Al = sp.lil_matrix(A)
        Ml = sp.lil_matrix(M)

        # Work on patches
        for dofs in dms:
            # Local matrices:
            
            try:  # Raises on UiO singularity 2017.1. image
                Aloc = Al[np.ix_(dofs,dofs)].todense()
                Mloc = Ml[np.ix_(dofs,dofs)].todense()
            except AttributeError:
                Aloc = np.array([[Al[np.ix_(dofs,dofs)]]])
                Mloc = np.array([[Ml[np.ix_(dofs,dofs)]]])
            
            # solve local eigenvalue problem:
            lam, Uloc = la.eigh(Aloc, b=Mloc, type=1)
            Uloc = np.asarray( Uloc )
            # Insert appropriately:
            B[np.ix_(dofs,dofs)] += np.dot( Uloc * (lam**(-s)), Uloc.T)

        # Set matrix:
        self.B = sp.csr_matrix(B)

    def __call__(self, b):
        res = np.zeros(self.B.shape[1])
        mask = self.mask
        res[mask] = self.eta * self.B[np.ix_(mask,mask)].dot(b[mask])
        return res
