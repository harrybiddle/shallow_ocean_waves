import argparse
import logging
import math
from random import randint, random
import sys

from humanfriendly import parse_timespan, parse_length
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

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
#  dU/dT = - gravity * dH/dX - drag * U
#  dV/dT = - gravity * dH/dY - drag * V
#  dH/dT = - ( dU/dX + dV/dY ) * Hbackground / dX

ONE_MILLISECOND = 1

def create_grids(ni, nj):
    u = np.zeros((nj + 2, ni + 3))
    v = np.zeros((nj + 3, ni + 2))
    h = np.zeros((nj + 2, ni + 2))
    speed = np.zeros((nj, ni)) # note that speed doesn't have ghost values
    return u, v, h, speed

def create_central_bump(nj, ni, width=0.25):
    ''' Creates a grid with a small cosine-like 'bump' in the centre. The height
    is 1 and the diameter of the base is a WIDTH fraction of the grid width. For
    example, WIDTH=0.25 corresponds to a quarter of the grid width.
    '''
    def wave_shape(x):
        ''' A wave with unit height at X=0, going down to zero at X=width/2 '''
        x = np.clip(x, a_min=0, a_max=width / 2)
        y = math.cos(2 * math.pi * x / width) + 1
        return y / 2 # normalise to [0, 1]

    def normalised_distance_to_grid_centre(j, i):
        d = lambda i, j: math.sqrt((i - ni / 2) ** 2 + (j - nj / 2) ** 2)
        return d(i, j) / d(0, 0)

    bump = np.zeros((nj, ni))
    for j in range(0, nj):
        for i in range(0, ni):
            d = normalised_distance_to_grid_centre(j, i)
            bump[j, i] = wave_shape(d)
    return bump

class RandomDropper():
    ''' Given a height field, can be used to add water 'drops' (i.e.
    cosine-like waves) at random locations '''

    def __init__(self, h):
        self.h = h[1:-1, 1:-1]
        self.nj, self.ni = self.h.shape
        self.bump = create_central_bump(self.nj, self.ni)

    def add_drop_at_random_location(self):
        random_shift = (randint(0, self.nj), randint(0, self.nj))
        random_bump = np.roll(self.bump, random_shift, axis=(0, 1))
        self.h += random_bump

def reflect_boundary(array, right_boundary=1, bottom_boundary=1):
    ''' Fills ghost cells with reflected values of the non-ghost cells. Ghost
    cells are in a boundary of a width 1 on the left and top, and RIGHT_BOUNDARY
    and BOTTOM_BOUNDARY on the right and bottom respectively '''
    rb, bb = right_boundary, bottom_boundary
    interior = array[1:-bb, 1:-rb]

    array[0, 0]     = interior[-1, -1]               # top left cell
    array[1:-bb, 0] = interior[:, -1]                # left column
    array[0, 1:-rb] = interior[-1, :]                # top row
    for i in range(1, rb + 1):
        array[1:-bb, -i] = interior[:, rb - i]       # right column(s)
        array[0, -i]     = interior[-1, rb - i]      # top right cell(s)
    for j in range(1, bb + 1):
        array[-j, 1:-rb] = interior[bb - j, :]       # bottom rows(s)
        array[-j, 0]     = interior[bb - j, -1]      # bottom left cell(s)
        for i in range(1, rb + 1):
            array[-j, -i] = interior[bb - j, rb - i]           # bottom right cell(s)

def reflect_ghost_cells(u, v, h):
    reflect_boundary(u, right_boundary=2)
    reflect_boundary(v, bottom_boundary=2)
    reflect_boundary(h)

def compute_time_derivatives(u, v, h, c):
    ''' According to the equations:

            du/dt = - gravity * dh/dx - drag * u
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
    du_dt = - c.gravity * dh_dx - c.drag * u[:, 1:-1]
    dv_dt = - c.gravity * dh_dy - c.drag * v[1:-1, :]
    dh_dt = - (du_dx + dv_dy) * c.h_background / c.dx

    return du_dt, dv_dt, dh_dt

def apply_time_derivatives(u, v, h, du_dt, dv_dt, dh_dt, dt):
    u[:, 1:-1] += du_dt * dt
    v[1:-1, :] += dv_dt * dt
    h += dh_dt * dt

def timestep(u, v, h, dt, constants):
    ''' Meat of the simulation: progress u, v and h forwards in time by a
    quantity dt using forward Euler '''
    reflect_ghost_cells(u, v, h)
    du_dt, dv_dt, dh_dt = compute_time_derivatives(u, v, h, constants)
    apply_time_derivatives(u, v, h, du_dt, dv_dt, dh_dt, dt)

def compute_speed(u, v, speed):
    u_cell_centered = (u[1:-1, 1:-2] + u[1:-1, 2:-1]) * 0.5
    v_cell_centered = (v[1:-2, 1:-1] + v[2:-1, 1:-1]) * 0.5
    np.copyto(dst=speed, src=np.sqrt(u_cell_centered**2 + v_cell_centered ** 2))

class AdapativeTwoStep():

    def __init__(self, u, v, h, speed, dropper, constants,
                 t=0, epsilon=1e-5, max_steps=1000):
        ''' Implements the simple Euler 2-Step Adaptive Step Size algorithm
        from http://www.math.ubc.ca/~feldman/math/vble.pdf. '''
        # ingest all arguments to self
        for name, value in vars().items():
            if name != 'self':
                setattr(self, name, value)

        # take one frame to be the starting dt
        self.dt = constants.seconds_per_frame

        # create temporary copies for the substeps
        self.u1 = np.array(u, copy=True)
        self.v1 = np.array(v, copy=True)
        self.h1 = np.array(h, copy=True)
        self.u2 = np.array(u, copy=True)
        self.v2 = np.array(v, copy=True)
        self.h2 = np.array(h, copy=True)

    def _debug_log(self, dt, E, r, error_below_threshold):
        logging.debug('dt={}, E={}, r={}, sufficient accuracy? {}'.format(
            dt, E, r, error_below_threshold))

    def _timestep(self):
        ''' Move the simulation one timestep forwards. We will initially attempt
        to step forward by an amount self.dt, but if the error is too large,
        self.dt will be reduced until it is below the required threshold '''

        steps = 0
        dt = self.dt
        while True:
            # bail out if we're not converging
            if steps > self.max_steps:
                raise RuntimeError('Not converged after {} steps'.format(steps))
            steps += 1

            # reset temporary arrays
            np.copyto(dst=self.u1, src=self.u)
            np.copyto(dst=self.v1, src=self.v)
            np.copyto(dst=self.h1, src=self.h)
            np.copyto(dst=self.u2, src=self.u)
            np.copyto(dst=self.v2, src=self.v)
            np.copyto(dst=self.h2, src=self.h)

            # take one step forwards by dt
            timestep(self.u1, self.v1, self.h1, dt, self.constants)

            # take two half-steps forwards by dt/2
            timestep(self.u2, self.v2, self.h2, dt / 2, self.constants)
            timestep(self.u2, self.v2, self.h2, dt / 2, self.constants)

            # compare the two in order to generate an error estimate
            # only comparing height field here: could go further and compare u
            # and v too, but this seems to be OK
            E = np.linalg.norm(self.h2[1:-1, 1:-1] - self.h1[1:-1, 1:-1],
                               ord=np.inf)
            r = E / dt

            # break out if the error is accceptible
            error_below_threshold = (r < self.epsilon)
            self._debug_log(dt, E, r, error_below_threshold)
            if error_below_threshold:
                self.dt = dt
                self.t += dt
                # combine the two solutions to get the lowest error possible
                # TODO replace with a swap for speed
                np.copyto(dst=self.u, src=(2 * self.u2 - self.u1))
                np.copyto(dst=self.v, src=(2 * self.v2 - self.v1))
                np.copyto(dst=self.h, src=(2 * self.h2 - self.h1))
                return steps

            # repeat with a reduced dt if the error is too high
            dt = 0.9 * self.epsilon * dt / r
            logging.debug ('Reduced dt to {}'.format(dt))

    def step_to_next_frame(self):
        # possibly add in a new water drop
        if random() < self.constants.drop_probability:
            self.dropper.add_drop_at_random_location()

        # step until we are past the target time. will overstep a bit, but it
        # doesn't make much of a visual difference
        target_time = self.t + self.constants.seconds_per_frame
        total_steps = 0
        while self.t < target_time:
            steps = self._timestep()
            total_steps += steps

        # compute the speed
        compute_speed(self.u, self.v, self.speed)

        print ('time = {:.2f}s in {} timesteps'.format(self.t, total_steps))

class Video():
    ''' Given a image and a callback function that mutates it to the next frame,
    displays the frames as a video in a QT app '''

    def __init__(self, pixels, max_pixel, progress_frame):
        ''' Arguments:
            pixels: a 2D array whose entries represent pixel intensity.
            progress_frame: a function taking no arguments that updates the
                image to the next frame.
        '''
        self.pixels = pixels
        self.progress_frame = progress_frame
        colours = [(0,114,255), (14,210,247)]
        self.cmap = pg.ColorMap(pos=[0, max_pixel], color=colours)

    def _create_qt_application(self):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()
        self.image = pg.ImageItem()
        self.view = self.win.addViewBox()
        self.view.addItem(self.image)
        nj, ni = self.pixels.shape
        self.view.setRange(QtCore.QRectF(0, 0, nj, ni))

    def _progress_frame_and_update_image(self):
        self.progress_frame()
        coloured_pixels = self.cmap.map(self.pixels)
        self.image.setImage(coloured_pixels)
        QtCore.QTimer.singleShot(ONE_MILLISECOND,
                                 self._progress_frame_and_update_image)

    def _start_qt_event_loop(self):
        self.app.exec_()

    def show(self):
        ''' Show the video: a blocking call '''
        self._create_qt_application()
        self._progress_frame_and_update_image()
        self._start_qt_event_loop()

def parse_args(argv):
    ''' Parse an array of command-line options into a argparse.Namespace '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--ni', type=int, default=200)
    parser.add_argument('--nj', type=int, default=200)
    parser.add_argument('-n', type=int)
    parser.add_argument('--drag', type=float, default=1.E-6)
    parser.add_argument('--gravity', type=float, default=9.8e-4)
    parser.add_argument('--width', type=float, default=100000)
    parser.add_argument('--height', type=float, default=100000)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--depth', default='4km')
    parser.add_argument('--time-per-frame', default='1 hour')
    parser.add_argument('--drop-probability', type=float, default=1e-2)
    parser.add_argument('--maximum-speed', type=float, default=0.003)
    parser.add_argument('-v', '--debug', action='store_true')
    args = parser.parse_args(argv[1:])

    # parse time units
    args.seconds_per_frame = parse_timespan(args.time_per_frame)
    args.h_background = parse_length(args.depth)

    # pick --n option over --ni and --nj, if supplied
    if args.n is not None:
        args.ni = args.nj = args.n

    # set dx and dy
    args.dx = args.width / args.ni
    args.dy = args.width / args.nj

    return args

def main(argv):
    args = parse_args(argv)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # starting arrays and initial conditions
    u, v, h, speed = create_grids(args.ni, args.nj)

    # create an object that adds water drops. since the shape of the drop
    # involves cosines and is a bit slow, we use this object to cache it
    dropper = RandomDropper(h)

    # create an initial drop to get things going
    dropper.add_drop_at_random_location()

    # create timestepper object, which will be used to progress the simulation
    timestepper = AdapativeTwoStep(u, v, h, speed, dropper, args)

    # create a video of the simulation
    Video(pixels=speed, progress_frame=timestepper.step_to_next_frame,
          max_pixel=args.maximum_speed).show()

if __name__ == '__main__':
    main(sys.argv)