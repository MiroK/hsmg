from dolfin import PETScMatrix, as_backend_type, IndexMap
from contextlib import contextmanager
from scipy.sparse import csr_matrix
from petsc4py import PETSc


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
def petsc_serial_matrix(test_space, trial_space):
    '''PETsc.Mat from trial_space to test_space to be filled in...'''
    mesh = test_space.mesh()
    comm = mesh.mpi_comm().tompi4py()
    assert comm.size == 1
    
    # Alloc
    mat = PETSc.Mat().create(comm)

    row_map = test_space.dofmap()
    col_map = trial_space.dofmap()
    
    mat.setSizes([[row_map.index_map().size(IndexMap.MapSize_OWNED),
                   row_map.index_map().size(IndexMap.MapSize_GLOBAL)],
                  [col_map.index_map().size(IndexMap.MapSize_OWNED),
                   col_map.index_map().size(IndexMap.MapSize_GLOBAL)]])
    mat.setType('aij')
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
