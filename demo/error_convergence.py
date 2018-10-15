#from fenics_ii.utils.norms import H1_L2_InterpolationNorm
from dolfin import errornorm, interpolate
import numpy as np


def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        cr.next()
        return cr
    return start


@coroutine
def monitor_error(u, norms, memory, reduction=lambda x: x):
    mesh_size0, error0 = None, None
    while True:
        uh = yield
        mesh_size = uh[0].function_space().mesh().hmin()

        error = [norm(ui, uhi) for norm, ui, uhi in zip(norms, u, uh)]
        error = np.array(reduction(error))

        if error0 is not None:
            rate = np.log(error/error0)/np.log(mesh_size/mesh_size0)
        else:
            rate = np.nan*np.ones_like(error)
            
        print 'h = %.4E' % mesh_size,
        print ' ,'.join(['e_(u%d) = %.2E[%.2f]' % (i, e, r)
                         for i, (e, r) in enumerate(zip(error, rate))])
        
        error0, mesh_size0 = error, mesh_size
        memory.append(np.r_[mesh_size, error])

        
H1_norm = lambda u, uh: errornorm(u, uh, 'H1', degree_rise=1)
H10_norm = lambda u, uh: errornorm(u, uh, 'H10', degree_rise=1)
L2_norm = lambda u, uh: errornorm(u, uh, 'L2', degree_rise=1)
Hdiv_norm = lambda u, uh: errornorm(u, uh, 'Hdiv', degree_rise=1)
Hdiv0_norm = lambda u, uh: errornorm(u, uh, 'Hdiv0', degree_rise=1)
Hcurl_norm = lambda u, uh: errornorm(u, uh, 'Hcurl', degree_rise=1)
Hcurl0_norm = lambda u, uh: errornorm(u, uh, 'Hcurl0', degree_rise=1)


def Hs_norm(s):
    '''Computes the spectral Hs norm on the space mesh as uh.'''
    # So no degree rise
    def foo(u, uh):
        V = uh.function_space()
        u = interpolate(u, V)
        e = u.vector().get_local() - uh.vector().get_local()
        
        norm = H1_L2_InterpolationNorm(V)
        mat = norm.get_s_norm(s, as_type=np.ndarray)

        return np.sqrt(np.inner(e, mat.dot(e)))
    return foo
    
# --------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import Expression, UnitIntervalMesh, FunctionSpace, interpolate
    
    f = Expression('sin(pi*x[0])', degree=4)
    g = Expression('sin(4*pi*x[0])', degree=4)

    memory = []
    monitor = monitor_error([f, g], [L2_norm, H1_norm], memory)

    for n in [2, 4, 8, 16, 32]:
        mesh = UnitIntervalMesh(n)
        V = FunctionSpace(mesh, 'CG', 1)
        u = interpolate(f, V)
        v = interpolate(g, V)

        monitor.send([u, v])
    print memory
