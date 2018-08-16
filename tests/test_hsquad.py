from dolfin import *
from hsmg.hsquad import FLaplace, FHelmholtz
from hsmg.utils import from_np_array
import numpy as np

import unittest


class TestHsNorm(unittest.TestCase):
    def test_def(self):
        '''Approximation of truth'''
        k = 2
        f = Expression('cos(k*pi*x[0])', k=k, degree=4)

        # Solve (-Delta + I)^s u_exact = f
        for s in (0.25, 0.5, 0.75, -0.25):
            hs, errors = [], []
            u_exact = Expression('cos(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)
            
            for n in [8, 16, 32, 64, 128]:
                mesh = UnitIntervalMesh(n)

                V = FunctionSpace(mesh, 'CG', 1)

                b = assemble(inner(TestFunction(V), f)*dx)
                # Numeric
                H = FHelmholtz(V, s, bcs=None, compute_k=0.5)
                # NOTE: so the map is nodal to dual
                uh = Function(V, H*b)

                error = errornorm(u_exact, uh, 'L2')
                hs.append(mesh.hmin())
                errors.append(error)
            # Decrease
            self.assertTrue(all(np.diff(errors) < 0))
            # Order
            deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            self.assertTrue(deg > 1)

            
class TestHs0Norm(unittest.TestCase):
    def test_def(self):
        '''Approximation of truth'''
        k = 2
        f = Expression('sin(k*pi*x[0])', k=k, degree=4)

        # Solve (-Delta + I)^s u_exact = f
        for s in (0.25, 0.5, 0.75, -0.25):
            hs, errors = [], []
            u_exact = Expression('sin(k*pi*x[0])/pow(pow(k*pi, 2) + 1, s)', s=s, k=k, degree=4)
            
            for n in [8, 16, 32, 64, 128]:
                mesh = UnitIntervalMesh(n)

                V = FunctionSpace(mesh, 'CG', 1)

                b = assemble(inner(TestFunction(V), f)*dx)
                # Numeric
                bcs = DirichletBC(V, Constant(0), 'on_boundary')
                H = FLaplace(V, s, bcs, compute_k=0.5)
                # NOTE: so the map is nodal to dual
                uh = Function(V, H*b)

                error = errornorm(u_exact, uh, 'L2')
                hs.append(mesh.hmin())
                errors.append(error)
            # Decrease
            self.assertTrue(all(np.diff(errors) < 0))
            # Order
            deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            self.assertTrue(deg > 1)
