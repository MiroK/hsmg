from dolfin import MeshFunction, SubsetIterator, CompiledSubDomain
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import plot, sqrt

from fenics_ii.trace_tools.embedded_mesh import EmbeddedMesh
from block.iterative import MinRes, CGN
import numpy as np
import os


def log_results(args, size, results, name='', fmt='%.18e'):
    '''Cols of size -> result'''
    
    nrows = len(size)
    assert all(nrows == len(results[s]) for s in results)
    offset = len(size[0])

    with open(args.log, 'w') as handle:
        for fract in sorted(results.keys()):
            data = results[fract]
            
            table = np.zeros((nrows, offset + len(data[0])))
            for row, (s, d) in enumerate(zip(size, data)):
                table[row, :offset] = s
                table[row, offset:] = d

            # snorm*
            if 's' in args:
                args.s = fract  # Instead of range
            header = map(lambda (k, v): '%s: %s' % (k, str(v)), args.__dict__.iteritems())

            # EMI
            if 's' not in args:
                header.append('%s: %s' % ('s', fract))
            
            header = ', '.join([name] + header)

            np.savetxt(handle, table, fmt=fmt, header=header)

    return args.log


def iter_solve((AA, bb, BB), tolerance):
    '''MinRes solve to get iteration counts'''
    x = AA.create_vec()
    x.randomize()

    def monitor(k, x, r):
        print k, r
    
    AAinv = MinRes(AA, precond=BB, initial_guess=x,
                   tolerance=tolerance, relativeconv=True, maxiter=500, show=2,
                   callback=monitor)

    # Compute solution
    x = AAinv * bb

    niters = len(AAinv.residuals) - 1
    size = [xi.size() for xi in x]
    
    return size, niters


def cond_solve((AA, bb, BB), tolerance):
    '''Solve CGN to get the estimate of condition number'''
    x = AA.create_vec()
    x.randomize()

    AAinv = CGN(AA, precond=BB, initial_guess=x,
                tolerance=tolerance, relativeconv=True, maxiter=1500, show=2)

    # Compute solution
    x = AAinv * bb

    niters = len(AAinv.residuals) - 1
    size = [xi.size() for xi in x]

    lmin, lmax = np.sort(np.abs(AAinv.eigenvalue_estimates()))[[0, -1]]
    cond = sqrt(lmax/lmin)
    
    return size, cond


def compute_hierarchy(dim, n, nlevels):
    '''
    The mesh where we want to solve is n. Here we compute previous
    levels for setting up multrid. nlevels in total.
    '''
    assert nlevels > 0

    if len(dim) == 1:
        xd = None
        yd = int(dim)
    # No intersection
    elif len(dim) == 2:
        curve_dict = __NOT_ISECT__
        xd = int(dim[0])
        yd = int(dim[1])
    else:
        # Loop or with isect
        if dim[0] == '0':
            curve_dict = __LOOPS__
        else:
            curve_dict = __ISECT__
        xd = int(dim[1])
        yd = int(dim[2])

    if nlevels == 1:
        # Background
        mesh_init = {1: UnitIntervalMesh,
                     2: UnitSquareMesh,
                     3: UnitCubeMesh}[yd]

        gamma = None if xd is None else CompiledSubDomain(curve_dict[dim])
        
        mesh = mesh_init(*(n, )*yd)
        if gamma is None: return [mesh]

        markers = MeshFunction('size_t', mesh, xd, 0)
        gamma.mark(markers, 1)

        # plot(markers, interactive=True, hide_below=0.5)

        assert sum(1 for _ in SubsetIterator(markers, 1)) > 0
        # NOTE: !(EmbeddedMesh <:  Mesh)
        return [EmbeddedMesh(mesh, markers, 1).mesh]

    return compute_hierarchy(dim, n, 1) + compute_hierarchy(dim, n/2, nlevels-1)


__LOOPS__ = {'012': 'near(std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)), 0.25)',
             '023': 'near(std::max(fabs(x[0] - 0.5), std::max(fabs(x[1] - 0.5), fabs(x[2] - 0.5))), 0.25)',
             '013': '''(near(x[0]*(1-x[0]), 0.0) && near(x[1]*(1-x[1]), 0.0)) ||
                       (near(x[1]*(1-x[1]), 0.0) && near(x[2]*(1-x[2]), 0.0)) ||
                       (near(x[2]*(1-x[2]), 0.0) && near(x[0]*(1-x[0]), 0.0))'''}

__ISECT__ = {'-12': 'near(x[0], 0) || near (x[1], 1)',
             '-23': 'near(x[0], 0) || near (x[1], 1)',
             '-13': '''(near(x[2], 0) && near(x[0], 0)) ||
                       (near(x[2], 0) && near(x[1], 0)) ||
                       (near(x[0], 0) && near(x[1], 0))'''}

__NOT_ISECT__ = {'12': 'near(x[0], x[1])',
                 '23': 'near(x[0], x[1])',
                 '13': 'near(x[0], x[1]) && near(x[1], x[2])'}
