import argparse
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
#        --- i --->
#      ................................
#      .       .       .       .       .
#  |   .  h00  .  h01  .  h02  .  h03  .
#  j   .       .       .       .       .
#  |   ........-----------------........
#  V   .       |       |       |       .
#      .  h10  |  h11  |  h12  |  h13  .
#      .       |       |       |       .
#      ........-----------------........
#      .       |       |       |       .
#      .  h20  |  h21  |  h22  |  h23  .
#      .       |       |       |       .
#      ........-----------------........
#      .       .       .       .       .
#      .  h30  .  h31  .  h32  .  h33  .
#      .       .       .       .       .  shape (nj + 2, ni + 2)
#     .................................
#
#
#        --- i --->
#      ..~v00~...~v01~...~v02~...~v03~..
#      .       .       .       .       .
#  |   .       .       .       .       .
#  j   .       .       .       .       .
#  |   ..~v10~.---v11-----v12---.~v13~..
#  V   .       |       |       |       .
#      .       |       |       |       .
#      .       |       |       |       .
#      ..~v20~.---v21-----v22---.~v23~..
#      .       |       |       |       .
#      .       |       |       |       .
#      .       |       |       |       .
#      ..~v30~.--~v31~---~v32~--.~v33~..
#      .       .       .       .       .
#      .       .       .       .       .
#      .       .       .       .       .
#      ..~v40~...~v41~...~v42~...~v43~..  shape (nj + 3, ni + 2)
#
#
#        --- i --->
#      .................................
#      .       .       .       .       .
#  | ~u00~   ~u01~   ~u02~   ~u03~   ~u04~
#  j   .       .       .       .       .
#  |   ........-----------------........
#  v   .       |       |       |       .
#    ~u10~    u11     u12    ~u13~   ~u14~
#      .       |       |       |       .
#      ........-----------------........
#      .       |       |       |       .
#    ~u20~    u21     u22    ~u23~   ~u24~
#      .       |       |       |       .
#      ........-----------------........
#      .       .       .       .       .
#    ~u30~   ~u31~   ~u32~   ~u33~   ~u34~
#      .       .       .       .       .
#      .................................  shape (nj + 2, ni + 3)
#
#
#
#
# Governing equations:
#
#  dU/dT =   rotation * V - gravity * dH/dX - drag * U + wind
#  dV/dT = - rotation * U - gravity * dH/dY - drag * V
#  dH/dT = - ( dU/dX + dV/dY ) * Hbackground / dX
#
#
#

MILLISECONDS_PER_SECOND = 1000

def create_h(ni, nj):
    return np.zeros((nj + 2, ni + 2))

def create_u(ni, nj):
    return np.zeros((nj + 2, ni + 3))

def create_v(ni, nj):
    return np.zeros((nj + 3, ni + 2))

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


def compute_du_dt(h, u, v, rotation, drag, gravity, wind, dx):
    ''' According to the equation:
            du/dt = - gravity * dh/dx - drag * u + wind
    Return is without ghost values '''
    dh_dx = np.diff(h, axis=1) / dx
    return - gravity * dh_dx - drag * u[:, 1:-1] + wind

def compute_dv_dt(h, u, v, rotation, drag, gravity, dy):
    ''' According to the equation:
            dv/dt = - gravity * dh/dy - drag * v
    Return is without ghost values '''
    dh_dy = np.diff(h, axis=0) / dy
    return - gravity * dh_dy - drag * v[1:-1, :]

def compute_dh_dt(u, v, h_background, dx):
    ''' According to the equation:
            dh/dt = - ( du/dx + dv/dy ) * h_background / dx
    '''
    du_dx = np.diff(u, axis=1)
    dv_dy = np.diff(v, axis=0)
    return - (du_dx + dv_dy) * h_background / dx

def timestep_u(u, du_dt, dt):
    u[:, 1:-1] += du_dt * dt

def timestep_v(v, dv_dt, dt):
    v[1:-1, :] += dv_dt * dt

def timestep_h(h, dh_dt, dt):
    h += dh_dt * dt

def reflect_ghost_cells(array, right_boundary=1, bottom_boundary=1):
    ''' Fills ghost cells with reflected values of the non-ghost cells. Ghost
    cells are in a boundary of a width 1 on the left and top, and RIGHT_BOUNDARY
    and BOTTOM_BOUNDARY on the right and bottom respectively '''
    non_ghost_cells = array[1:-bottom_boundary, 1:-right_boundary]
    t = np.tile(non_ghost_cells, (2, 2))
    t = np.roll(t, 1, axis=0)
    t = np.roll(t, 1, axis=1)
    nj, ni = array.shape
    np.copyto(dst=array, src=t[0:nj, 0:ni])

def reflect_u_ghost_cells(u):
    return reflect_ghost_cells(u, right_boundary=2)

def reflect_v_ghost_cells(v):
    return reflect_ghost_cells(v, bottom_boundary=2)

def reflect_h_ghost_cells(h):
    return reflect_ghost_cells(h)

def timestep(u, v, h, dt, constants):
    reflect_u_ghost_cells(u)
    reflect_v_ghost_cells(v)
    reflect_h_ghost_cells(h)

    c = constants
    du_dt = compute_du_dt(h, u, v, c.rotation, c.drag, c.gravity, c.wind, c.dx)
    dv_dt = compute_dv_dt(h, u, v, c.rotation, c.drag, c.gravity, c.dy)
    dh_dt = compute_dh_dt(u, v, c.h_background, c.dx)

    timestep_u(u, du_dt, dt)
    timestep_v(v, dv_dt, dt)
    timestep_h(h, dh_dt, dt)

def simulate_and_draw_frame(frame_number, simulate_frame, plot_surface):
    ''' Animation function to be passed to matplotlib.animation.FuncAnimation.
    Progresses the simulation forwards and then updates the image.

    Arguments:
        frame_number: (unused) the number of the frame that should be updated
        simulate_frame: a function that progresses a simulation to the next frame.
            it takes no arguments and has no return value.
        plot_surface: a function that creates a new surface image. It takes
            no arguments but returns the matplotlib artist.

    Returns:
        An iterable of artists that matplotlib should update
    '''
    simulate_frame()
    return [plot_surface()]

def clear_axes_and_plot_surface(axes, x, y, h):
    ''' Plot the current height grid as a 3D surface. The z-axis is hard-coded
    to [0, 1] regardless of the height values.

    Arguments:
        axes: a matplotlib.Axes3D object. It will be clear()'d on every call,
            so don't set any options on it!
        x, y: arrays of the x and y coordinates. These arrays should be the same
            size as the height field array without ghost values.
        h: height field, including ghost values
    '''
    axes.clear()
    axes.set_zlim(0, 1)
    clipped = h[1:-1, 1:-1]
    return axes.plot_surface(x, y, clipped, cmap=cm.coolwarm, vmin=0, vmax=1)

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

    def step_forwards(self):
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
            self.timestep(self.u1, self.v1, self.h1, dt)

            # take two half-steps forwards by dt/2
            self.timestep(self.u2, self.v2, self.h2, dt / 2)
            self.timestep(self.u2, self.v2, self.h2, dt / 2)

            # compare the two in order to generate an error estimate
            # only comparing height field here: could go further and compare u
            # and v too, but this seems to be OK
            E = np.linalg.norm(self.h2[1:-1, 1:-1] - self.h1[1:-1, 1:-1],
                               ord=np.inf)
            r = E / dt
            print ('E=', E, 'r=', r, 'Acceptable?', r < self.epsilon)

            # break out if the error is accceptible
            error_below_threshold = (r < self.epsilon)
            if error_below_threshold:
                self.dt = dt
                self.t += dt
                # combine the two solutions to get the lowest error possible
                # TODO replace with a swap for speed
                np.copyto(dst=self.u, src=(2 * self.u2 - self.u1))
                np.copyto(dst=self.v, src=(2 * self.v2 - self.v1))
                np.copyto(dst=self.h, src=(2 * self.h2 - self.h1))
                return steps, self.dt

            # repeat with a reduced dt if the error is too high
            dt_new = 0.9 * self.epsilon * dt / r
            print (dt, ' is reduced to ', dt_new)
            dt = dt_new

    def step_to_next_frame(self):
        # step until we are past the target time. will overstep a bit, but it
        # doesn't make much of a visual difference
        target_time = self.t + self.seconds_per_frame
        total_steps = 0
        while self.t < target_time:
            steps, _ = self.step_forwards()
            total_steps += steps

        print ('time = {:.2f}s in {} timesteps'.format(self.t, total_steps))

def parse_args(argv):
    ''' Parse an array of command-line options into a argparse.Namespace '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--ni', type=int, default=200)
    parser.add_argument('--nj', type=int, default=200)
    parser.add_argument('--n', type=int)
    parser.add_argument('--rotation', type=float, default=0.0)
    parser.add_argument('--drag', type=float, default=1.E-6)
    parser.add_argument('--gravity', type=float, default=9.8e-4)
    parser.add_argument('--wind', type=float, default=1.e-8)
    parser.add_argument('--dx', type=float, default=10.E3)
    parser.add_argument('--dy', type=float, default=10.E3)
    parser.add_argument('--dt', type=float, default=60, dest='target_dt')
    parser.add_argument('--h_background', type=float, default=4000)
    parser.add_argument('--speed-multiplier', type=int, default=70000)
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args(argv[1:])

    # pick --n option over --ni and --nj, if supplied
    if args.n is not None:
        args.ni = args.nj = args.n

    return args

def main(argv):
    args = parse_args(argv)

    # starting arrays and initial conditions
    u = create_u(args.ni, args.nj)
    v = create_v(args.ni, args.nj)
    h = create_h(args.ni, args.nj)
    create_bump_in_centre(h)

    # create initial plot
    x = np.linspace(0, 100, args.ni)
    y = np.linspace(0, 100, args.nj)
    x, y = np.meshgrid(x, y)
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    clear_axes_and_plot_surface(axes, x, y, h)

    # create timestepper object that encapsulates how to progress the simulation
    # forwards in time
    seconds_per_frame = args.speed_multiplier / args.fps
    timestep_function = lambda u, v, h, dt: timestep(u, v, h, dt, args)
    timestepper = Timestepper(u, v, h, timestep_function, seconds_per_frame)

    # create loop to update plot and progress simulation forwards
    millseconds_per_frame = MILLISECONDS_PER_SECOND / args.fps
    _ = animation.FuncAnimation(figure,
            simulate_and_draw_frame,
            fargs=(lambda: clear_axes_and_plot_surface(axes, x, y, h),
                   lambda: timestepper.step_to_next_frame()),
            interval=millseconds_per_frame)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)