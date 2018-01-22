from dolfin import Expression
from sympy.printing import ccode
import sympy as sp


def expr_body(expr, **kwargs):
    if not hasattr(expr, '__len__'):
        # Defined in terms of some coordinates
        xyz = set(sp.symbols('x[0], x[1], x[2]'))
        xyz_used = xyz & expr.free_symbols
        assert xyz_used <= xyz
        # Expression params which need default values
        params = (expr.free_symbols - xyz_used) & set(kwargs.keys())
        # Body
        expr = ccode(expr).replace('M_PI', 'pi')
        # Default to zero
        kwargs.update(dict((str(p), 0.) for p in params))
        # Convert
        return expr
    # Vectors, Matrices as iterables of expressions
    else:
        return [expr_body(e, **kwargs) for e in expr]


def as_expression(expr, degree=4, **kwargs):
    '''Turns sympy expressions to Dolfin expressions.'''
    return Expression(expr_body(expr), degree=degree, **kwargs)


def babuska_H1_2d():
    '''
    Exact solution for -Delta u + u = f on [0, 1]^2, Tu = g on boundary.
    '''
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    p = sp.S(0)  # Normal stress

    f = -u.diff(x, 2) - u.diff(y, 2) + u
    g = u

    up = map(as_expression, (u, p))
    fg = map(as_expression, (f, g))

    return up, fg


def babuska_H1_3d():
    '''
    Exact solution for -Delta u + u = f on [0, 1]^3, Tu = g on boundary.
    '''
    x, y, z = sp.symbols('x[0], x[1], x[2]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y)*z*(1-z))
    p = sp.S(0)  # Normal stress

    f = -u.diff(x, 2) - u.diff(y, 2) - u.diff(z, 2) + u
    g = u

    up = map(as_expression, (u, p))
    fg = map(as_expression, (f, g))

    return up, fg


def babuska_Hdiv_2d():
    '''
    Exact solution for -Delta u + u = f on [0, 1]^2, grad(u).n = g on boundary.
    The mixed form. With sigma = grad(u) and -u = p on the boundary
    '''
    x, y  = sp.symbols('x[0], x[1]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y))
    sigma = (u.diff(x, 1), u.diff(y, 1))
    p = -u

    f = -u.diff(x, 2) - u.diff(y, 2) + u
    g = sp.S(0)

    up = map(as_expression, (sigma, u, p))
    fg = map(as_expression, (f, g))

    return up, fg


def babuska_Hdiv_3d():
    '''
    Exact solution for -Delta u + u = f on [0, 1]^3, grad(u).n = g on boundary.
    The mixed form. With sigma = grad(u) and -u = p on the boundary
    '''
    x, y, z = sp.symbols('x[0], x[1], x[2]')
    u = sp.cos(sp.pi*x*(1-x)*y*(1-y)*z*(1-z))
    sigma = (u.diff(x, 1), u.diff(y, 1), u.diff(z, 1))
    p = -u

    f = -u.diff(x, 2) - u.diff(y, 2) - u.diff(z, 2) + u
    g = sp.S(0)

    up = map(as_expression, (sigma, u, p))
    fg = map(as_expression, (f, g))

    return up, fg


def grad_div_2d():
    '''
    -grad(div(sigma)) + sigma = f in [0, 1]^2
                      sigma.n = g on the boundary

    To be solved with Lagrange multiplier to enforce bcs rather then
    enforcing them on the function space level.
    '''

    x, y = sp.symbols('x[0] x[1]')

    sigma = sp.Matrix([sp.sin(sp.pi*x*(1-x)*y*(1-y)),
                       sp.sin(2*sp.pi*x*(1-x)*y*(1-y))])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)

    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    f = -sp_grad(sp_div(sigma)) + sigma
    g = sp.S(0)

    sigma_exact = as_expression(sigma)
    # It's quite nice that you get surface divergence as the extra var
    p_exact = as_expression(sp_div(-sigma)) 
    f_rhs, g_rhs = map(as_expression, (f, g))

    return (sigma_exact, p_exact), (f_rhs, g_rhs)


def curl_curl_2d():
    '''
    rot(curl(sigma)) + sigma = f in [0, 1]^2
                     sigma.t = g on the boundary

    To be solved with Lagrange multiplier to enforce bcs rather then
    enforcing them on the function space level.
    '''

    x, y = sp.symbols('x[0] x[1]')

    sigma = sp.Matrix([sp.sin(sp.pi*x*(1-x)*y*(1-y)),
                       sp.sin(2*sp.pi*x*(1-x)*y*(1-y))])


    sp_grad = lambda f: sp.Matrix([f.diff(x, 1), f.diff(y, 1)])

    sp_div = lambda f: f[0].diff(x, 1) + f[1].diff(y, 1)

    # This is a consistent with FEniCS definition
    ROT_MAT = sp.Matrix([[sp.S(0), sp.S(1)], [sp.S(-1), sp.S(0)]])

    # Maps vector to scalar: 
    sp_curl = lambda f: sp_div(ROT_MAT*f)

    # Maps scalar to vector
    sp_rot = lambda f: ROT_MAT*sp_grad(f)

    f = sp_rot(sp_curl(sigma)) + sigma
    g = sp.S(0)

    sigma_exact = as_expression(sigma)
    # It's quite nice that you get surface curl as the extra var
    p_exact = as_expression(sp_curl(sigma))
    f_rhs, g_rhs = map(as_expression, (f, g))

    return (sigma_exact, p_exact), (f_rhs, g_rhs)
