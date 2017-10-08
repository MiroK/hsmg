from dolfin import UnitIntervalMesh, FunctionSpace, DomainBoundary
from dolfin import interpolate, Expression
from hsmg import Hs0NormMG

def test():
    mesh = UnitIntervalMesh(128)
    V = FunctionSpace(mesh, 'CG', 1)
    bdry = DomainBoundary()
    s = 0.5
    mg_params = {'macro_size': 1,
                 'nlevels': 4}

    MG_Half = Hs0NormMG(V, bdry, s, mg_params)
    
    x = interpolate(Expression('sin(k*pi*x[0])', k=3, degree=4), V).vector()
    # Action
    y = MG_Half*x
    # Nothing raises
