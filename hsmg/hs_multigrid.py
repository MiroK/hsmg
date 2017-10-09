# Return instance of Trygve's H^s multigrid. Its __call__ is an
# action of numpy array

def setup(A, M, R, bdry_dofs, macro_dofmap):
    '''
    Factory function for creating instances of MG methods for fractional Sobolev norms.
    INPUT:
        A: Stiffness matrices
        M: Mass matrices
        R: Restriction matrices
        bdry_dofs: Boundary DOFs on each mesh level in hierarchy.
        macro_dofmap: For each level tabulates DOFs for each subdomain, to be used in smoother.
    OUTPUT:
        res: Returns instance of MG-method that works on vectors, and can be called via __call__(b).
    '''
    # Dummy for testing
    class FracLapMG(object):
        '''Class for MG method''' 
        def __init__(self, A, M, R, bdry_dofs, macro_dofmap):
            '''Constructor'''
            raise NotImplementedError
        def multigrid_level(self,j,bj):
            '''Multigrid method on general level.
            INPUT:
                j: integer; level in mesh hierarchy
                bj: vector to apply MG method to.
            RETURN:
                res: Output vector from method.
            '''
            raise NotImplementedError

        def coarse_solve(self, bj):
            '''Solve exactly on coarsest level of mesh hierarchy.'''
        def __call__(self, b):
            '''Call method.'''
            return self.multigrid_level(0,b)

    return FracLapMG(A,M,R, bdry_dofs, macro_dofmap)
