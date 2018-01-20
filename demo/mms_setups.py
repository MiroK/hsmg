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
