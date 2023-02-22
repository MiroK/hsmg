from dolfin import PETScMatrix, as_backend_type, IndexMap, info
from contextlib import contextmanager
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import numpy as np

# NOTE: On my machine scipy eigh routines don't take advantage of threads
# and so eigvanlue computations for GEVP are way too slope. On the other 
# hand numpy's EVP solvers are fully threaded. Therefore it may pay off to
# solve GEVP as EVP by numpy. THat is
#
# A U = B U L is
# B^{-0.5} A U = B^{0.5} U L and by ansatz V = B^{-0.5} V we get sym EVP
# B^{-0.5} A B^{-0.5} V = V L
#
# So eigenvalues are same.
def my_eigvalsh(A, B):
    '''Au = lmbda Bu transforming to EVP'''
    # Transformation
    beta, U = np.linalg.eigh(B)
    info('\tDone power %g' % np.min(np.abs(beta)))
    Bnh = U.dot(np.diag(beta**-0.5).dot(U.T))
    # Eigenvalus of B^{-0.5} A B^{-0.5}
    S = Bnh.dot(A.dot(Bnh))
    return np.linalg.eigvalsh(S)


def my_eigh(A, B):
    '''Au = lmbda Bu transforming to EVP'''
    # Transformation
    beta, U = np.linalg.eigh(B)
    info('\tDone power %g' % np.min(np.abs(beta)))
    Bnh = U.dot(np.diag(beta**-0.5).dot(U.T))
    
    S = Bnh.dot(A.dot(Bnh))
    lmbda, V = np.linalg.eigh(S)
    info('\tDone S power %g' % np.min(np.abs(lmbda)))    
    # With transformed eigenvectors
    return lmbda, Bnh.dot(V)



def to_csr_matrix(A):
    '''Convert PETScMatrix/PETSc.Mat to scipy.csr'''
    if isinstance(A, PETSc.Mat):
        return csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

    return to_csr_matrix(as_backend_type(A).mat())


def transpose_matrix(A):
    '''Create a transpose of PETScMatrix/PETSc.Mat'''
    if isinstance(A, PETSc.Mat):
        At = PETSc.Mat()  # Alloc
        A.transpose(At)  # Transpose to At
        return At

    At = transpose_matrix(as_backend_type(A).mat())
    return PETScMatrix(At)


def from_csr_matrix(A, out=PETScMatrix):
    '''Create PETSc.Mat/PETScMatrix from csr_matrix'''
    B = PETSc.Mat().createAIJ(size=A.shape,
                              csr=(A.indptr, A.indices, A.data))
    if isinstance(B, out):
        return B
    else:
        return out(B)


def from_np_array(A, out=PETScMatrix):
    '''Create PETSc.Mat/PETScMatrix fom numpy array'''
    B = PETSc.Mat().createDense(size=A.shape, array=A)
    if isinstance(B, out):
        return B
    else:
        return out(B)


    
@contextmanager
def petsc_serial_matrix(test_space, trial_space, nnz=None):
    '''PETsc.Mat from trial_space to test_space to be filled in...'''
    mesh = test_space.mesh()
    comm = mesh.mpi_comm().tompi4py()
    assert comm.size == 1

    row_map = test_space.dofmap()
    col_map = trial_space.dofmap()
    
    sizes = [[row_map.index_map().size(IndexMap.MapSize_OWNED),
              row_map.index_map().size(IndexMap.MapSize_GLOBAL)],
             [col_map.index_map().size(IndexMap.MapSize_OWNED),
              col_map.index_map().size(IndexMap.MapSize_GLOBAL)]]
    
    # Alloc
    mat = PETSc.Mat().createAIJ(sizes, nnz=nnz, comm=comm)
    mat.setUp()
    
    row_lgmap = PETSc.LGMap().create(map(int, row_map.tabulate_local_to_global_dofs()),
                                     comm=comm)
    col_lgmap = PETSc.LGMap().create(map(int, col_map.tabulate_local_to_global_dofs()),
                                     comm=comm)
    mat.setLGMap(row_lgmap, col_lgmap)

    mat.assemblyBegin()
    # Fill
    yield mat
    # Tear down
    mat.assemblyEnd()
