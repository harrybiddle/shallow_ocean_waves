import argparse
import logging
import math
import sys

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D # required for 3D plots
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# make it the number of simulation cells
# dictate: cell centered, staggered i/j, etc.
# concept of worldspace and interpolation
#
# This is a cell centered grid, wrapping
# around in Y, zero at X:
#
#   (0, 0)  --- i --->               (0, 4)
#      ...~v~.....~v~.....~v~.....~v~...
#  |    .       .       .       .       .
#  j  ~u~ ~h~ ~u~ ~h~ ~u~ ~h~ ~u~ ~h~ ~u~
#  |   .       .       .       .       .
#  v   ...~v~..----v-------v----..~v~..
#      .       |       |       |       .
#     ~u~ ~h~  u   h   u   h  ~u~ ~h~  ~u~
#      .       |       |       |       .
#      ...~v~..|---v ---.--v --|..~v~...
#      .       |       |       |       .
#     ~u~ ~h~  u   h   u   h  ~u~ ~h~ ~u~
#      .       |       |       |       .
#      ...~v~..---~v~-----~v~---..~v~...
#      .       .       .       .       .
#     ~u~ ~h~  ~u~~h~ ~u~ ~h~ ~u~ ~h~ ~u~      h is shape (nj + 2, ni + 2)
#      .       .       .       .       .       u is shape (nj + 2, ni + 3)
#      ...~v~.....~v~.....~v~.....~v~...       v is shape (nj + 3, ni + 2)
#   (1, 0)                           (4, 4)
#
# Governing equations:
#
#  dU/dT =   rotation * V - gravity * dH/dX - drag * U + wind
#  dV/dT = - rotation * U - gravity * dH/dY - drag * V
#  dH/dT = - ( dU/dX + dV/dY ) * Hbackground / dX

MILLISECONDS_PER_SECOND = 1000

def create_grids(ni, nj):
    u = np.zeros((nj + 2, ni + 3))
    v = np.zeros((nj + 3, ni + 2))
    h = np.zeros((nj + 2, ni + 2))
    return u, v, h

def create_bump_in_centre(h, width=0.25):
    ''' Adds a small wave in the centre of the grid. The wave is of height one
    and the diameter of the base is a WIDTH fraction of the grid width. For
    example, WIDTH=0.25 corresponds to a quarter of the grid width.
    '''

    def wave_shape(x, width):
        ''' A wave with unit height at X=0, going down to zero at X=width/2 '''
        x = np.clip(x, a_min=0, a_max=width / 2)
        y = math.cos(2 * math.pi * x / width) + 1
        return y / 2 # normalise to [0, 1]

    nj, ni = h.shape
    def normalised_distance_to_grid_centre(i, j):
        d = lambda i, j: math.sqrt((i - ni / 2) ** 2 + (j - nj / 2) ** 2)
        return d(i, j) / d(0, 0)

    for j in range(0, nj):
        for i in range(0, ni):
            d = normalised_distance_to_grid_centre(i, j)
            h[j, i] = wave_shape(d, width)


def compute_time_derivatives(u, v, h, c):
    ''' According to the equations:

            du/dt = - gravity * dh/dx - drag * u + wind
            dv/dt = - gravity * dh/dy - drag * v
            dh/dt = - (du/dx + dv/dy) * h_background / dx

    Returns arrays are created new and returned without any ghost values.
    '''

    # spatial derivatives
    dh_dx = np.diff(h, axis=1) / c.dx
    dh_dy = np.diff(h, axis=0) / c.dy
    du_dx = np.diff(u, axis=1) / c.dx
    dv_dy = np.diff(v, axis=0) / c.dy

    # construct time derivatives
    du_dt = - c.gravity * dh_dx - c.drag * u[:, 1:-1] + c.wind
    dv_dt = - c.gravity * dh_dy - c.drag * v[1:-1, :]
    dh_dt = - (du_dx + dv_dy) * c.h_background / c.dx

    return du_dt, dv_dt, dh_dt

def apply_time_derivatives(u, v, h, du_dt, dv_dt, dh_dt, dt):
    u[:, 1:-1] += du_dt * dt
    v[1:-1, :] += dv_dt * dt
    h += dh_dt * dt

def reflect_boundary(array, right_boundary=1, bottom_boundary=1):
    ''' Fills ghost cells with reflected values of the non-ghost cells. Ghost
    cells are in a boundary of a width 1 on the left and top, and RIGHT_BOUNDARY
    and BOTTOM_BOUNDARY on the right and bottom respectively '''
    non_ghost_cells = array[1:-bottom_boundary, 1:-right_boundary]
    t = np.tile(non_ghost_cells, (2, 2))
    t = np.roll(t, 1, axis=0)
    t = np.roll(t, 1, axis=1)
    nj, ni = array.shape
    np.copyto(dst=array, src=t[0:nj, 0:ni])

def reflect_ghost_cells(u, v, h):
    reflect_boundary(u, right_boundary=2)
    reflect_boundary(v, bottom_boundary=2)
    reflect_boundary(h)

def timestep(u, v, h, dt, constants):
    np.copyto(dst=h, src=np.roll(h, 1, axis=1))

class Timestepper():

    def __init__(self, u, v, h, timestep, seconds_per_frame, t=0, epsilon=1e-5,
                 max_steps=1000):
        ''' Implements the simple Euler 2-Step Adaptive Step Size algorithm
        from http://www.math.ubc.ca/~feldman/math/vble.pdf.
        '''
        # ingest all arguments to self
        for name, value in vars().items():
            if name != 'self':
                setattr(self, name, value)

        # take one frame to be the starting dt
        self.dt = seconds_per_frame

        # create temporary copies for the substeps
        self.u1 = np.array(u, copy=True)
        self.v1 = np.array(v, copy=True)
        self.h1 = np.array(h, copy=True)
        self.u2 = np.array(u, copy=True)
        self.v2 = np.array(v, copy=True)
        self.h2 = np.array(h, copy=True)

    def debug_log(self, dt, E, r, error_below_threshold):
        logging.debug(('dt={}, E={}, r={}, sufficient accuracy? {}'
                ).format(dt, E, r, error_below_threshold))

    def step_forwards(self):
        self.timestep(self.u, self.v, self.h, self.dt)
        self.t += self.dt
        return 1, self.dt

    total_frames = 0

    def step_to_next_frame(self):
        # step until we are past the target time. will overstep a bit, but it
        # doesn't make much of a visual difference
        target_time = self.t + self.seconds_per_frame
        total_steps = 0
        while self.t < target_time:
            steps, _ = self.step_forwards()
            total_steps += steps
        self.total_frames += 1

        print ('time = {:.2f}s in {} timesteps'.format(self.t, total_steps))
        print (self.total_frames)


def main_loop(_, image, h, timestepper):
    timestepper.step_to_next_frame()
    image.set_data(h)
    return [image]

def parse_args(argv):
    ''' Parse an array of command-line options into a argparse.Namespace '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--ni', type=int, default=200)
    parser.add_argument('--nj', type=int, default=200)
    parser.add_argument('--n', type=int)
    #parser.add_argument('--rotation', type=float, default=0.0)
    parser.add_argument('--drag', type=float, default=1.E-6)
    parser.add_argument('--gravity', type=float, default=9.8e-4)
    parser.add_argument('--wind', type=float, default=1.e-8)
    parser.add_argument('--dx', type=float, default=500)
    parser.add_argument('--dy', type=float, default=500)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--h_background', type=float, default=4000)
    parser.add_argument('--speed-multiplier', type=int, default=60000)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('-v', '--debug', action='store_true')
    args = parser.parse_args(argv[1:])

    # pick --n option over --ni and --nj, if supplied
    if args.n is not None:
        args.ni = args.nj = args.n

    return args

def main(argv):
    args = parse_args(argv)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # starting arrays and initial conditions
    u, v, h = create_grids(args.ni, args.nj)
    create_bump_in_centre(h)

    # create timestepper object. this is used to progress the simulation
    seconds_per_frame = args.speed_multiplier / args.fps
    timestep_function = lambda u, v, h, dt: timestep(u, v, h, dt, args)
    timestepper = Timestepper(u, v, h, timestep_function, seconds_per_frame)

    # initial plot
    image = plt.imshow(h, cmap='hot')

    # create loop to progress the simulation and updates the plot
    millseconds_per_frame = MILLISECONDS_PER_SECOND / args.fps
    _ = animation.FuncAnimation(plt.gcf(),
            main_loop, fargs=(image, h, timestepper),
            interval=1)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)