import math
import unittest
from unittest import mock

import numpy as np

from simulation import *
from euler_two_step import *

class NumpyTestCase(unittest.TestCase):

    def create_new_test_array(self):
        return np.array([[1, 4, 8, 2, 4, 8, 1],
                         [4, 8, 1, 5, 8, 5, 5],
                         [2, 3, 0, 0, 1, 9, 1],
                         [8, 2, 1, 8, 0, 2, 4],
                         [1, 1, 7, 1, 5, 0, 5],
                         [2, 3, 0, 0, 7, 0, 7]])

    def assert_arrays_equal(self, a, b):
        self.assertTrue(np.array_equal(a, b))

    def assert_arrays_unequal(self, a, b):
        self.assertFalse(np.array_equal(a, b))

class TestAdaptiveTimeStepper(unittest.TestCase):

    def setUp(self):
        self.u = np.arange(25).reshape((5, 5))
        self.v = np.arange(25).reshape((5, 5))
        self.h = np.arange(25).reshape((5, 5))
        self.timestep = mock.Mock()
        self.starting_dt = 0.1
        self.epsilon = 0.1
        self.max_steps = 10
        self.timestepper = AdaptiveTimestepper(self.u, self.v, self.h,
                                               self.timestep, self.starting_dt,
                                               epsilon=self.epsilon,
                                               max_steps=self.max_steps)

    def test_start_time(self):
        self.assertEqual(0, self.timestepper.time())

    def test_progress_with_low_error(self):
        steps, dt = self.timestepper.step_forwards()
        self.assertEqual(1, steps)
        self.assertEqual(self.starting_dt, dt)
        self.timestep.assert_has_calls([
            mock.call(mock.ANY, mock.ANY, mock.ANY, 0.1),
            mock.call(mock.ANY, mock.ANY, mock.ANY, 0.05),
            mock.call(mock.ANY, mock.ANY, mock.ANY, 0.05)])

    # def test_timestepper(self):

    #     a  = AdaptiveTimestepper(u, v, h, timestep, starting_dt,
    #                              epsilon=1):
    #     f0 = 0
    #     f = mock.Mock()
    #     a = AdaptiveTimestepper(f0, f, starting_h=0.1, epsilon=0.1)

    #     a.step_forwards()

# class TestStartingArrays(NumpyTestCase):

#     def test_centered_grid_shape(self):
#         u = create_h(10, 10)
#         self.assertEqual((12, 12), u.shape)

#     def test_staggered_x_grid_shape(self):
#         u = create_u(10, 10)
#         self.assertEqual((12, 13), u.shape)

#     def test_staggered_y_grid_shape(self):
#         v = create_v(10, 10)
#         self.assertEqual((13, 12), v.shape)

#     # def test_starting_height_is_column(self):
#     #     h = np.arange(81).reshape(9, 9)
#     #     add_central_column(h)
#     #     v = 0
#     #     for j in range(0, 9):
#     #         for i in range(0, 9):
#     #             if (3 <= i and i <= 5) and (3 <= j and j <= 5):
#     #                 self.assertEqual(v + 1, h[j][i])
#     #             else:
#     #                 self.assertEqual(v + 0, h[j][i])
#     #             v += 1

# class TestCompute(NumpyTestCase):

#     def setUp(self):
#         self.random_h = np.random.rand(4, 5)
#         self.random_u = np.random.rand(4, 6)
#         self.random_v = np.random.rand(5, 5)

#         self.random_h_copy = np.array(self.random_h, copy=True)
#         self.random_u_copy = np.array(self.random_u, copy=True)
#         self.random_v_copy = np.array(self.random_v, copy=True)

#     def assert_arrays_unchanged(self):
#         self.assert_arrays_equal(self.random_h_copy, self.random_h)
#         self.assert_arrays_equal(self.random_u_copy, self.random_u)
#         self.assert_arrays_equal(self.random_v_copy, self.random_v)

#     def test_compute_du_dt_values(self):
#         u = np.arange(48).reshape(6, 8)
#         h = self.create_new_test_array()
#         h_copy = np.array(h, copy=True)
#         u_copy = np.array(u, copy=True)
#         du_dt = compute_du_dt(h, u, v=None,
#                               rotation=0, drag=1, gravity=1, wind=0, dx=1, dt=1)

#         expected = -np.array([[ 4,  6, -3,  6, 9, -1],
#                               [13,  3, 15, 15, 10, 14],
#                               [18, 15, 19, 21, 29, 14],
#                               [19, 25, 34, 20, 31, 32],
#                               [33, 40, 29, 40, 32, 43],
#                               [42, 39, 43, 51, 38, 53]])

#         self.assert_arrays_equal(u, u_copy)
#         self.assert_arrays_equal(h, h_copy)
#         self.assert_arrays_equal(du_dt, expected)

#     def test_compute_du_dt_shape(self):
#         du_dt = compute_du_dt(self.random_h, self.random_u, self.random_v,
#                               rotation=.2, drag=-1.5, gravity=2., wind=.1,
#                               dx=1.9, dt=1)
#         self.assertEqual(du_dt.shape[0], self.random_u.shape[0])
#         self.assertEqual(du_dt.shape[1], self.random_u.shape[1] - 2)

#     def test_compute_du_dt_does_not_change_input(self):
#         compute_du_dt(self.random_h, self.random_u, self.random_v,
#                       rotation=.2, drag=-1.5, gravity=2., wind=.1, dx=1.9, dt=1)
#         self.assert_arrays_unchanged()

#     def test_compute_dv_dt_shape(self):
#         dv_dt = compute_dv_dt(self.random_h, self.random_u, self.random_v,
#                               rotation=.2, drag=-1.5, gravity=22/7, dy=1.9, dt=1)
#         self.assertEqual(dv_dt.shape[0], self.random_v.shape[0] - 2)
#         self.assertEqual(dv_dt.shape[1], self.random_v.shape[1])

#     def test_compute_dv_dt_does_not_change_input(self):
#         dv_dt = compute_dv_dt(self.random_h, self.random_u, self.random_v,
#                               rotation=.2, drag=-1.5, gravity=22/7, dy=1.9, dt=1)
#         self.assert_arrays_unchanged()

#     def test_compute_dh_dt_shape(self):
#         dh_dt = compute_dh_dt(self.random_u, self.random_v,
#                               h_background=100.2, dx=1.9, dt=1)
#         self.assertEqual(dh_dt.shape, self.random_h.shape)

#     def test_compute_dh_dt_does_not_change_input(self):
#         dh_dt = compute_dh_dt(self.random_u, self.random_v,
#                               h_background=100.2, dx=1.9, dt=1)
#         self.assert_arrays_unchanged()

# class TestBoundary(NumpyTestCase):
#     ''' Check that boundary reflection functions are working. In the expected
#     arrays below ghost values are written in parentheses for clarity '''

#     def test_u_ghost_cells(self):
#         u = self.create_new_test_array()
#         reflect_u_ghost_cells(u)
#         expected = np.array([[(5), (1), (7), (1), (5), (1), (7)],
#                              [(8),   8,   1,   5,   8, (8), (1)],
#                              [(1),   3,   0,   0,   1, (3), (0)],
#                              [(0),   2,   1,   8,   0, (2), (1)],
#                              [(5),   1,   7,   1,   5, (1), (7)],
#                              [(8), (8), (1), (5), (8), (8), (1)]])
#         self.assert_arrays_equal(u, expected)

#     def test_v_ghost_cells(self):
#         v = self.create_new_test_array()
#         reflect_v_ghost_cells(v)
#         expected = np.array([[(2), (2), (1), (8), (0), (2), (2)],
#                              [(5),   8,   1,   5,   8,   5, (8)],
#                              [(9),   3,   0,   0,   1,   9, (3)],
#                              [(2),   2,   1,   8,   0,   2, (2)],
#                              [(5), (8), (1), (5), (8), (5), (8)],
#                              [(9), (3), (0), (0), (1), (9), (3)]])
#         self.assert_arrays_equal(v, expected)

#     def test_h_ghost_cells(self):
#         h = self.create_new_test_array()
#         reflect_h_ghost_cells(h)
#         expected = np.array([[(0), (1), (7), (1), (5), (0), (1)],
#                              [(5),   8,   1,   5,   8,   5, (8)],
#                              [(9),   3,   0,   0,   1,   9, (3)],
#                              [(2),   2,   1,   8,   0,   2, (2)],
#                              [(0),   1,   7,   1,   5,   0, (1)],
#                              [(5), (8), (1), (5), (8), (5), (8)]])
#         self.assert_arrays_equal(h, expected)

# class ParseArgs(unittest.TestCase):

#     def check_ni_nj(self, ni=None, nj=None, n=None):
#         argv = [None]
#         if ni is not None:
#             argv.extend(['--ni', str(ni)])
#         if nj is not None:
#             argv.extend(['--nj', str(nj)])
#         if n is not None:
#             argv.extend(['--n', str(n)])
#         args = parse_args(argv)
#         return args.ni, args.nj

#     def test_n_overides_ni_and_nj_options(self):
#         self.assertEqual((10, 12), self.check_ni_nj(ni=10, nj=12))
#         self.assertEqual((4, 4), self.check_ni_nj(n=4, ni=10, nj=12))
#         self.assertEqual((4, 4), self.check_ni_nj(n=4))

# class TestSolverAgainstAnalyticalSolutions(NumpyTestCase):

#     def test_drag_component(self):
#         ''' Set up the following equations, which have analytical solutions
#         u = u_0 * exp(- c_drag * t):

#                 du_dt = - c_drag * u
#                 dv_dt = - c_drag * v

#         We can run a number of timesteps and check that our numerical solutions
#         are close.
#         '''

#         # set parameters
#         n = 10
#         dt = 0.001
#         nsteps = 250
#         drag = 0.8
#         u_0 = 1.5
#         v_0 = - 2.5

#         constants = parse_args([None,
#                                '--n', str(n),
#                                '--rotation', '0',
#                                '--gravity', '0',
#                                '--drag', str(drag),
#                                 '--wind', '0'])

#         # initial conditions
#         u = create_u(n, n)
#         v = create_v(n, n)
#         h = create_h(n, n)
#         u[:] = u_0
#         v[:] = v_0

#         # solve for nsteps
#         for _ in range(0, nsteps):
#             timestep(u, v, h, dt, constants)

#         # check every cell is equal to the analytical solution
#         analytical_solution_u = u_0 * math.exp( - drag * nsteps * dt)
#         analytical_solution_v = v_0 * math.exp( - drag * nsteps * dt)

#         inner_u = u[1:-1, 1:-2]
#         inner_v = v[1:-2, 1:-1]

#         self.assertTrue(np.all(inner_u - analytical_solution_u < 0.01))
#         self.assertTrue(np.all(inner_v - analytical_solution_v < 0.01))

# class TestTimestepper(unittest.TestCase):

#     def test_timestepper_easiest_case(self):
#         f = mock.Mock()
#         t = Timestepper(target_dt=1,
#                         fps=1,
#                         simulation_seconds_per_video_second=1,
#                         timestep_function=f)
#         t.step_to_next_frame()
#         f.assert_called_once_with(1.0)

#     def test_timestepper_fps_large_dt(self):
#         f = mock.Mock()
#         t = Timestepper(target_dt=1,
#                         fps=24,
#                         simulation_seconds_per_video_second=1,
#                         timestep_function=f)
#         t.step_to_next_frame()
#         f.assert_called_once_with(1/24)

#     def test_timestepper_fps_small_dt_double_speed(self):
#         f = mock.Mock()
#         t = Timestepper(target_dt=1/40,
#                         fps=24,
#                         simulation_seconds_per_video_second=2,
#                         timestep_function=f)
#         t.step_to_next_frame()
#         self.assertEqual(4, f.call_count)
#         f.assert_has_calls([mock.call(1/48)] * 4)

if __name__ == '__main__':
    unittest.main()