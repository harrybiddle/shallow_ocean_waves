import math
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from simulation import *

def create_new_test_array():
    return np.array([[1, 4, 8, 2, 4, 8, 1],
                     [4, 8, 1, 5, 8, 5, 5],
                     [2, 3, 0, 0, 1, 9, 1],
                     [8, 2, 1, 8, 0, 2, 4],
                     [1, 1, 7, 1, 5, 0, 5],
                     [2, 3, 0, 0, 7, 0, 7]])

class TestStartingArrays(unittest.TestCase):

    def test_grid_shapes(self):
        u, v, h, speed = create_grids(10, 10)
        self.assertEqual((12, 12), h.shape)
        self.assertEqual((12, 13), u.shape)
        self.assertEqual((13, 12), v.shape)
        self.assertEqual((10, 10), speed.shape)


class TestTimeDerivatives(unittest.TestCase):

    def setUp(self):
        self.rand_h = np.random.rand(6, 7)
        self.rand_u = np.random.rand(6, 8)
        self.rand_v = np.random.rand(7, 7)

    def compute_time_derivatives(self, u=np.zeros((6, 8)), v=np.zeros((7, 7)),
                                 h=np.zeros((6, 7)), gravity=0.0, dx=1.0,
                                 dy=1.0, drag=0.0, h_background=1.0):
        constants = SimpleNamespace(gravity=gravity, dx=dx, dy=dy, drag=drag,
                                    h_background=h_background)
        return compute_time_derivatives(u, v, h, constants)

    def test_dh_dx_term(self):
        ''' Create a situation where du_dt = dh_dx * gravity, in order to
        test this term '''
        h = create_new_test_array()
        gravity = 2
        du_dt, _, _ = self.compute_time_derivatives(h=h, gravity=gravity)
        expected = np.array([[ -3,  -4,   6,  -2,  -4,   7],
                             [ -4,   7,  -4,  -3,   3,   0],
                             [ -1,   3,   0,  -1,  -8,   8],
                             [  6,   1,  -7,   8,  -2,  -2],
                             [  0,  -6,   6,  -4,   5,  -5],
                             [ -1,   3,   0,  -7,   7,  -7]]) * gravity
        np.testing.assert_equal(du_dt, expected)

    def test_compute_time_derivatives_does_not_change_input(self):
        rand_h_copy = np.array(self.rand_h, copy=True)
        rand_u_copy = np.array(self.rand_u, copy=True)
        rand_v_copy = np.array(self.rand_v, copy=True)

        self.compute_time_derivatives(u=self.rand_u, v=self.rand_v,
                                      h=self.rand_h, gravity=1.0, drag=1.0)

        np.testing.assert_equal(rand_h_copy, self.rand_h)
        np.testing.assert_equal(rand_u_copy, self.rand_u)
        np.testing.assert_equal(rand_v_copy, self.rand_v)

    def test_compute_time_derivatives_shape(self):
        r = self.compute_time_derivatives(u=self.rand_u, v=self.rand_v,
                                          h=self.rand_h, gravity=1.0, drag=1.0)
        du_dt, dv_dt, dh_dt = r
        self.assertEqual(du_dt.shape[0], self.rand_u.shape[0])
        self.assertEqual(du_dt.shape[1], self.rand_u.shape[1] - 2)
        self.assertEqual(dv_dt.shape[0], self.rand_v.shape[0] - 2)
        self.assertEqual(dv_dt.shape[1], self.rand_v.shape[1])
        self.assertEqual(dh_dt.shape, self.rand_h.shape)

class TestComputeCurl(unittest.TestCase):

    def test_compute_curl(self):

        dx = dy = 1.0
        u = create_new_test_array()[:-1, :]
        v = create_new_test_array()[:, :-1]

        u_at_v = (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[-1:, 1:]) * 0.25
        v_at_u = (v[:-1, :-1] + v[1:, :-1] + v[1:, 1:] + v[-1:, 1:]) * 0.25
        du_dy = np.diff(u_at_v, axis=0) / dy
        dv_dx = np.diff(v_at_u, axis=1) / dx
        curl = dv_dx[1:-1, :] - du_dy[:, 1:-1]

class TestBoundary(unittest.TestCase):
    ''' Check that boundary reflection functions are working. In the expected
    arrays below ghost values are written in parentheses for clarity '''

    def setUp(self):
        self.dummy = np.zeros((10, 10))

    def test_u_ghost_cells(self):
        u = create_new_test_array()
        reflect_ghost_cells(u, self.dummy, self.dummy)
        expected = np.array([[(5), (1), (7), (1), (5), (1), (7)],
                             [(8),   8,   1,   5,   8, (8), (1)],
                             [(1),   3,   0,   0,   1, (3), (0)],
                             [(0),   2,   1,   8,   0, (2), (1)],
                             [(5),   1,   7,   1,   5, (1), (7)],
                             [(8), (8), (1), (5), (8), (8), (1)]])
        np.testing.assert_equal(u, expected)

    def test_v_ghost_cells(self):
        v = create_new_test_array()
        reflect_ghost_cells(self.dummy, v, self.dummy)
        expected = np.array([[(2), (2), (1), (8), (0), (2), (2)],
                             [(5),   8,   1,   5,   8,   5, (8)],
                             [(9),   3,   0,   0,   1,   9, (3)],
                             [(2),   2,   1,   8,   0,   2, (2)],
                             [(5), (8), (1), (5), (8), (5), (8)],
                             [(9), (3), (0), (0), (1), (9), (3)]])
        np.testing.assert_equal(v, expected)

    def test_h_ghost_cells(self):
        h = create_new_test_array()
        reflect_ghost_cells(self.dummy, self.dummy, h)
        expected = np.array([[(0), (1), (7), (1), (5), (0), (1)],
                             [(5),   8,   1,   5,   8,   5, (8)],
                             [(9),   3,   0,   0,   1,   9, (3)],
                             [(2),   2,   1,   8,   0,   2, (2)],
                             [(0),   1,   7,   1,   5,   0, (1)],
                             [(5), (8), (1), (5), (8), (5), (8)]])
        np.testing.assert_equal(h, expected)

class TestArgumentParsing(unittest.TestCase):

    def check_ni_nj(self, ni=None, nj=None, n=None):
        argv = [None]
        if ni is not None:
            argv.extend(['--ni', str(ni)])
        if nj is not None:
            argv.extend(['--nj', str(nj)])
        if n is not None:
            argv.extend(['--n', str(n)])
        args = parse_args(argv)
        return args.ni, args.nj

    def test_n_overides_ni_and_nj_options(self):
        self.assertEqual((10, 12), self.check_ni_nj(ni=10, nj=12))
        self.assertEqual((4, 4), self.check_ni_nj(n=4, ni=10, nj=12))
        self.assertEqual((4, 4), self.check_ni_nj(n=4))

class TestSolverAgainstAnalyticalSolutions(unittest.TestCase):

    def test_drag_component(self):
        ''' Set up the following equations, which have analytical solutions
        u = u_0 * exp(- c_drag * t):

                du_dt = - c_drag * u
                dv_dt = - c_drag * v

        We can run a number of timesteps and check that our numerical solutions
        are close.
        '''

        # set parameters
        n = 10
        dt = 0.001
        nsteps = 250
        drag = 0.8
        u_0 = 1.5
        v_0 = - 2.5

        constants = parse_args([None,
                               '--n', str(n),
                               # '--rotation', '0',
                               '--gravity', '0',
                               '--drag', str(drag)])

        # initial conditions
        u, v, h, _ = create_grids(n, n)
        u[:] = u_0
        v[:] = v_0

        # solve for nsteps
        for _ in range(0, nsteps):
            timestep(u, v, h, dt, constants)

        # check every cell is equal to the analytical solution
        analytical_solution_u = u_0 * math.exp( - drag * nsteps * dt)
        analytical_solution_v = v_0 * math.exp( - drag * nsteps * dt)

        inner_u = u[1:-1, 1:-2]
        inner_v = v[1:-2, 1:-1]

        self.assertTrue(np.all(inner_u - analytical_solution_u < 0.01))
        self.assertTrue(np.all(inner_v - analytical_solution_v < 0.01))

if __name__ == '__main__':
    unittest.main()