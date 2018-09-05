from dolfin import *
from hsmg.hseig import Hs0Norm, HsNorm
from hsmg.utils import from_np_array
import numpy as np

import unittest


class TestHsNorm(unittest.TestCase):

    def test_identity(self):
        '''Inverse works'''
        mesh = UnitSquareMesh(5, 5)
        
        V = FunctionSpace(mesh, 'CG', 1)
        H = HsNorm(V, s=0.123)

        x = H.create_vec()
        x.set_local(np.random.rand(x.local_size()))
                    
        y = (H**-1)*(H*x)

        error = (x-y).norm('linf')
        self.assertTrue(error < 1E-13)

    def test_def_1(self):
        '''Agreas with Laplace for s=1'''
        mesh = UnitSquareMesh(5, 5)
        
        V = FunctionSpace(mesh, 'CG', 1)
        H = HsNorm(V, s=1.0)

        u, v = TrialFunction(V), TestFunction(V)
        A = assemble(inner(u, v)*dx + inner(grad(u), grad(v))*dx)

        x = H.create_vec()
        x.set_local(np.random.rand(x.local_size()))
                    
        y1 = H*x
        y2 = A*x
        
        error = (y1 - y2).norm('linf')
        self.assertTrue(error < 1E-13)

    def test_def_0(self):
        '''Agrees with identity for s=0'''
        mesh = UnitSquareMesh(5, 5)
        
        V = FunctionSpace(mesh, 'CG', 1)
        H = HsNorm(V, s=0.0)

        u, v = TrialFunction(V), TestFunction(V)
        A = assemble(inner(u, v)*dx)

        x = H.create_vec()
        x.set_local(np.random.rand(x.local_size()))
                    
        y1 = H*x
        y2 = A*x
        
        error = (y1 - y2).norm('linf')
        self.assertTrue(error < 1E-13)

    def test_mult(self):
        '''s0*s1 = s0 + s1 (well)'''
        mesh = UnitSquareMesh(5, 5)
        
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        M = inner(u, v)*dx
        
        H0 = HsNorm(V, s=0.0)
        H1 = HsNorm(V, s=1.0)

        invM = from_np_array(np.linalg.inv(assemble(M).array()))
        H = H0 * invM * H1
        
        A = HsNorm(V, s=0.0 + 1.0)

        x = H.create_vec()
        x.set_local(np.random.rand(x.local_size()))
                    
        y1 = H*x
        y2 = A*x
        
        error = (y1 - y2).norm('linf')
        self.assertTrue(error < 1E-13)


    def test_def(self):
        '''Approximation of truth'''
        for s in (0.25, 0.5, 0.75):
            hs, errors = [], []
            for n in [8, 16, 32, 64, 128, 256, 512]:
                mesh = UnitIntervalMesh(n)

                V = FunctionSpace(mesh, 'CG', 1)

                k = 2
                # An eigenfunction
                f = interpolate(Expression('cos(k*pi*x[0])', k=k, degree=1), V).vector()
                # Numeric
                H = HsNorm(V, s=s)
                Hs_norm = f.inner(H*f)
                # From def <(-Delta + I)^s u, v> 
                truth = ((k*pi)**2 + 1)**(s) * 0.5  # 0.5 is form L2 norm of f)

                error = abs(truth - Hs_norm)
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
        for s in (0.25, 0.5, 0.75):
            hs, errors = [], []
            for n in [8, 16, 32, 64, 128, 256, 512]:
                mesh = UnitIntervalMesh(n)

                V = FunctionSpace(mesh, 'CG', 1)

                k = 2
                # An eigenfunction
                f = interpolate(Expression('sin(k*pi*x[0])', k=k, degree=1), V).vector()
                # Numeric
                H = Hs0Norm(V, bcs=DirichletBC(V, Constant(0), 'on_boundary'), s=s)
                Hs_norm = f.inner(H*f)
                # From def <(-Delta + I)^s u, v> 
                truth = (k*pi)**(2*s) * 0.5  # 0.5 is form L2 norm of f)

                error = abs(truth - Hs_norm)
                hs.append(mesh.hmin())
                errors.append(error)
            # Decrease
            self.assertTrue(all(np.diff(errors) < 0))
            # Order
            deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            self.assertTrue(deg > 1)

    def test_def_nodal_dual(self):
        '''Approximation of truth'''
        for s in (0.25, 0.5, 0.75):
            hs, errors = [], []
            for n in [8, 16, 32, 64, 128, 256, 512]:
                mesh = UnitIntervalMesh(n)

                V = FunctionSpace(mesh, 'CG', 1)

                k = 2
                # An eigenfunction
                f = interpolate(Expression('sin(k*pi*x[0])', k=k, degree=1), V)
                # Numeric
                H = Hs0Norm(V, bcs=DirichletBC(V, Constant(0), 'on_boundary'), s=s)
                dHf = H*f.vector() # This is dual
                # Want nodal
                Hf = Function(V)
                u, v = TrialFunction(V), TestFunction(V)
                solve(assemble(inner(u, v)*dx), Hf.vector(), dHf)
                # From def <(-Delta + I)^s u, v> 
                truth = (k*pi)**(2*s)
                f = truth*f

                error = sqrt(abs(assemble(inner(f - Hf, f - Hf)*dx)))
                hs.append(mesh.hmin())
                errors.append(error)
            # Decrease
            self.assertTrue(all(np.diff(errors) < 0))
            # Order
            deg = np.polyfit(np.log(hs), np.log(errors), 1)[0]
            self.assertTrue(deg > 1)
