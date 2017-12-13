import math
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from simulation import *

class NumpyTestCase(unittest.TestCase):

    def create_new_test_array(self):
        return np.array([[1, 4, 8, 2, 4, 8, 1],
                         [4, 8, 1, 5, 8, 5, 5],
                         [2, 3, 0, 0, 1, 9, 1],
                         [8, 2, 1, 8, 0, 2, 4],
                         [1, 1, 7, 1, 5, 0, 5],
                         [2, 3, 0, 0, 7, 0, 7]])

    def assert_arrays_equal(self, a, b):
        np.testing.assert_equal(a, b)

    def assert_arrays_unequal(self, a, b):
        self.assertFalse(np.array_equal(a, b))

# class TestAdaptiveTimeStepper(unittest.TestCase):

#     def setUp(self):
#         self.u = np.arange(25).reshape((5, 5))
#         self.v = np.arange(25).reshape((5, 5))
#         self.h = np.arange(25).reshape((5, 5))
#         self.timestep = mock.Mock()
#         self.starting_dt = 0.1
#         self.epsilon = 0.1
#         self.max_steps = 10
#         self.timestepper = AdaptiveTimestepper(self.u, self.v, self.h,
#                                                self.timestep, self.starting_dt,
#                                                epsilon=self.epsilon,
#                                                max_steps=self.max_steps)

#     def test_start_time(self):
#         self.assertEqual(0, self.timestepper.time())

#     def test_progress_with_low_error(self):
#         steps, dt = self.timestepper.step_forwards()
#         self.assertEqual(1, steps)
#         self.assertEqual(self.starting_dt, dt)
#         self.timestep.assert_has_calls([
#             mock.call(mock.ANY, mock.ANY, mock.ANY, 0.1),
#             mock.call(mock.ANY, mock.ANY, mock.ANY, 0.05),
#             mock.call(mock.ANY, mock.ANY, mock.ANY, 0.05)])

#     def test_timestepper(self):

#         a  = AdaptiveTimestepper(u, v, h, timestep, starting_dt,
#                                  epsilon=1)
#         f0 = 0
#         f = mock.Mock()
#         a = AdaptiveTimestepper(f0, f, starting_h=0.1, epsilon=0.1)

#         a.step_forwards()

class TestStartingArrays(NumpyTestCase):

    def test_grid_shapes(self):
        u, v, h = create_grids(10, 10)
        self.assertEqual((12, 12), h.shape)
        self.assertEqual((12, 13), u.shape)
        self.assertEqual((13, 12), v.shape)

    def test_starting_height(self):
        h = np.zeros((100, 100))
        create_bump_in_centre(h)
        self.assertEqual(1, h[50, 50])


class TestCompute(NumpyTestCase):

    def setUp(self):
        self.rand_h = np.random.rand(6, 7)
        self.rand_u = np.random.rand(6, 8)
        self.rand_v = np.random.rand(7, 7)

        self.rand_h_copy = np.array(self.rand_h, copy=True)
        self.rand_u_copy = np.array(self.rand_u, copy=True)
        self.rand_v_copy = np.array(self.rand_v, copy=True)

        self.constants = SimpleNamespace(gravity=2.0, wind=1.0, dx=1.9, dy=1.9,
                                         drag=-1.5, h_background=100.2)

    def assert_arrays_unchanged(self):
        self.assert_arrays_equal(self.rand_h_copy, self.rand_h)
        self.assert_arrays_equal(self.rand_u_copy, self.rand_u)
        self.assert_arrays_equal(self.rand_v_copy, self.rand_v)

    # def test_values_of_du_dt(self):
    #     u = np.arange(48).reshape(6, 8)
    #     h = self.create_new_test_array()

    #     constants = SimpleNamespace(gravity=1.0, wind=0.0, dx=1, dy=1,
    #                                 drag=1.0, h_background=100.2)

    #     du_dt, _, _ = compute_time_derivatives(u, self.rand_v, h,
    #                                            self.constants)

    #     expected = -np.array([[ 4,  6, -3,  6, 9, -1],
    #                           [13,  3, 15, 15, 10, 14],
    #                           [18, 15, 19, 21, 29, 14],
    #                           [19, 25, 34, 20, 31, 32],
    #                           [33, 40, 29, 40, 32, 43],
    #                           [42, 39, 43, 51, 38, 53]])

    #     self.assert_arrays_equal(du_dt, expected)

    def test_compute_time_derivatives_does_not_change_input(self):
        compute_time_derivatives(self.rand_u, self.rand_v, self.rand_h,
                                 self.constants)
        self.assert_arrays_unchanged()

    def test_compute_time_derivatives_shape(self):
        r = compute_time_derivatives(self.rand_u, self.rand_v, self.rand_h,
                                     self.constants)
        du_dt, dv_dt, dh_dt = r
        self.assertEqual(du_dt.shape[0], self.rand_u.shape[0])
        self.assertEqual(du_dt.shape[1], self.rand_u.shape[1] - 2)
        self.assertEqual(dv_dt.shape[0], self.rand_v.shape[0] - 2)
        self.assertEqual(dv_dt.shape[1], self.rand_v.shape[1])
        self.assertEqual(dh_dt.shape, self.rand_h.shape)


class TestBoundary(NumpyTestCase):
    ''' Check that boundary reflection functions are working. In the expected
    arrays below ghost values are written in parentheses for clarity '''

    def setUp(self):
        self.dummy = np.zeros((10, 10))

    def test_u_ghost_cells(self):
        u = self.create_new_test_array()
        reflect_ghost_cells(u, self.dummy, self.dummy)
        expected = np.array([[(5), (1), (7), (1), (5), (1), (7)],
                             [(8),   8,   1,   5,   8, (8), (1)],
                             [(1),   3,   0,   0,   1, (3), (0)],
                             [(0),   2,   1,   8,   0, (2), (1)],
                             [(5),   1,   7,   1,   5, (1), (7)],
                             [(8), (8), (1), (5), (8), (8), (1)]])
        self.assert_arrays_equal(u, expected)

    def test_v_ghost_cells(self):
        v = self.create_new_test_array()
        reflect_ghost_cells(self.dummy, v, self.dummy)
        expected = np.array([[(2), (2), (1), (8), (0), (2), (2)],
                             [(5),   8,   1,   5,   8,   5, (8)],
                             [(9),   3,   0,   0,   1,   9, (3)],
                             [(2),   2,   1,   8,   0,   2, (2)],
                             [(5), (8), (1), (5), (8), (5), (8)],
                             [(9), (3), (0), (0), (1), (9), (3)]])
        self.assert_arrays_equal(v, expected)

    def test_h_ghost_cells(self):
        h = self.create_new_test_array()
        reflect_ghost_cells(self.dummy, self.dummy, h)
        expected = np.array([[(0), (1), (7), (1), (5), (0), (1)],
                             [(5),   8,   1,   5,   8,   5, (8)],
                             [(9),   3,   0,   0,   1,   9, (3)],
                             [(2),   2,   1,   8,   0,   2, (2)],
                             [(0),   1,   7,   1,   5,   0, (1)],
                             [(5), (8), (1), (5), (8), (5), (8)]])
        self.assert_arrays_equal(h, expected)

class ParseArgs(unittest.TestCase):

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

class TestSolverAgainstAnalyticalSolutions(NumpyTestCase):

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
                               '--drag', str(drag),
                                '--wind', '0'])

        # initial conditions
        u, v, h = create_grids(n, n)
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