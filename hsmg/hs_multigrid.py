# Return instance of Trygve's H^s multigrid. Its __call__ is an
# action of numpy array

def setup(A, M, R, bdry_dofs, macro_dofmap):
    '''TODO'''
    # Dummy for testing
    class Foo(object):
        def __call__(self, x):
            return A.dot(x)

    return Foo()
