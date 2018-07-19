from dolfin import (Timer, CompiledSubDomain, MeshFunction, UnitCubeMesh,
                    UnitSquareMesh, DomainBoundary, File)
from coarsen_2d import GmshCoarsener
from coarsen_1d import CurveCoarsenerIterative
from coarsen_common import smooth_manifolds
from xii import EmbeddedMesh


def square(n):
    mesh = UnitSquareMesh(n, n)
    f = MeshFunction('size_t', mesh, 1, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], x[1])').mark(f, 1)

    return EmbeddedMesh(f, 1)


def cube(n):
    mesh = UnitCubeMesh(n, n, n)
    f = MeshFunction('size_t', mesh, 2, 0)
    DomainBoundary().mark(f, 1)
    CompiledSubDomain('near(x[0], 0.5)').mark(f, 1)

    return EmbeddedMesh(f, 1)


def performance_1d():
    '''Scaling of coarsening curves'''
    domain = square
    
    ncells = []
    times = []
    for n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        mesh = domain(n)
        ncells.append(mesh.num_cells())

        t = Timer('coarsen')
        coarsener = CurveCoarsenerIterative
        _, success, after = coarsener.coarsen(mesh)

        times.append(t.stop())
    File('csquare.pvd') << after
    
    return ncells, times


def performance_2d():
    '''Scaling of coarsening manifolds'''
    domain = cube
    
    ncells = []
    times = []
    for n in (2, 4, 8, 16, 32):#, 64):
        mesh = domain(n)
        ncells.append(mesh.num_cells())

        t = Timer('coarsen')
        coarsener = GmshCoarsener('test.geo')
                
        _, success, after = coarsener.coarsen(mesh)

        times.append(t.stop())
    File('ccube.pvd') << after
    
    return ncells, times


def performance_manifolds(domain, cells):
    '''Scaling of manifold search'''
    ncells = []
    times = []
    for n in cells:
        mesh = domain(n)
        ncells.append(mesh.num_cells())

        t = Timer('coarsen')
        smooth_manifolds(mesh)
        times.append(t.stop())
    
    return ncells, times

# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    # ncells, times = performance_manifolds(square,
    #                                       [2**i for i in range(5, 11)])

    ncells, times = performance_manifolds(cube,
                                          [2**i for i in range(2, 7)])


    ncells = np.array(ncells)
    print 'O(%.2f)' % np.polyfit(np.log(ncells), np.log(times), deg=1)[0]

    plt.figure()
    plt.loglog(ncells, times, 'rx-', label='n')
    plt.loglog(ncells, (times[0]/ncells[0])*ncells, 'b:')
    plt.legend(loc='best')
    plt.show()

    # File('color_f.pvd') << f
