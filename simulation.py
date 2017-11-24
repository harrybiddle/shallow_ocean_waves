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

def create_h(ni, nj):
    return np.zeros((nj + 2, ni + 2))

def create_u(ni, nj):
    return np.zeros((nj + 2, ni + 3))

def create_v(ni, nj):
    return np.zeros((nj + 3, ni + 2))

def add_central_column(h):
    ''' Adds a column of height 1 in the middle third of the grid. Input
    is modified in-place '''
    nj, ni = h.shape
    ni_2 = ni / 2
    nj_2 = nj / 2

    def bump(x, width):
        x = np.clip(x, a_min=0, a_max=width / 2)
        y = math.cos(2 * math.pi * x / width) + 1
        return y / 2 # normalise to [0, 1]

    def dist(i, j):
        return math.sqrt((i - ni / 2) ** 2 + (j - nj / 2) ** 2)

    def norm_dist(i, j):
        return dist(i, j) / dist(0, 0)

    for j in range(0, nj):
        for i in range(0, ni):
            d = dist(i, j)
            h[j, i] = bump(norm_dist(i, j), 0.25)


def compute_du_dt(h, u, v, rotation, drag, gravity, wind, dx, dt):
    ''' According to the equation:
            du/dt = (- gravity * dh/dx - drag * u + wind) * dt
    Return is without ghost values '''
    dh_dx = np.diff(h, axis=1) / dx
    return (- gravity * dh_dx - drag * u[:, 1:-1] + wind) * dt

def compute_dv_dt(h, u, v, rotation, drag, gravity, dy, dt):
    ''' According to the equation:
            dv/dt = (- gravity * dh/dy - drag * v) * dt
    Return is without ghost values '''
    dh_dy = np.diff(h, axis=0) / dy
    return (- gravity * dh_dy - drag * v[1:-1, :]) * dt

def compute_dh_dt(u, v, h_background, dx, dt):
    ''' According to the equation:
            dh/dt = - (( du/dx + dv/dy ) * h_background / dx) * dt
    '''
    du_dx = np.diff(u, axis=1)
    dv_dy = np.diff(v, axis=0)
    return - ((du_dx + dv_dy) * h_background / dx) * dt

def timestep_u(u, du_dt):
    u[:, 1:-1] += du_dt

def timestep_v(v, dv_dt):
    v[1:-1, :] += dv_dt

def timestep_h(h, dh_dt):
    h += dh_dt

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

def timestep(u, v, h, constants):
    reflect_u_ghost_cells(u)
    reflect_v_ghost_cells(v)
    reflect_h_ghost_cells(h)

    c = constants
    du_dt = compute_du_dt(h, u, v, c.rotation, c.drag, c.gravity, c.wind, c.dx, c.dt)
    dv_dt = compute_dv_dt(h, u, v, c.rotation, c.drag, c.gravity, c.dy, c.dt)
    dh_dt = compute_dh_dt(u, v, c.h_background, c.dx, c.dt)

    timestep_u(u, du_dt)
    timestep_v(v, dv_dt)
    timestep_h(h, dh_dt)

def timestep_and_update_image(frame_number, dt, plot_surface, take_timestep):
    ''' Animation function to be passed to matplotlib.animation.FuncAnimation.
    Progresses the simulation forwards and then updates the image.

    Returns:
        An iterable of artists that matplotlib should update
    '''
    fps = 50
    for f in range(0, fps):
        print ('t = {:0.2f}'.format((fps * frame_number + f) * dt))
        take_timestep()
    return [plot_surface()]

def clear_axes_and_plot_surface(axes, x, y, h):
    axes.clear()
    axes.set_zlim(0, 1)
    clipped = h[1:-1, 1:-1]
    return axes.plot_surface(x, y, clipped, cmap=cm.coolwarm, vmin=0, vmax=1)

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ni', type=int, default=10)
    parser.add_argument('--nj', type=int, default=10)
    parser.add_argument('--n', type=int)
    parser.add_argument('--rotation', type=float, default=0.0)
    parser.add_argument('--drag', type=float, default=1.E-6)
    parser.add_argument('--gravity', type=float, default=9.8e-4)
    parser.add_argument('--wind', type=float, default=1.e-8)
    parser.add_argument('--dx', type=float, default=10.E3)
    parser.add_argument('--dy', type=float, default=10.E3)
    parser.add_argument('--dt', type=float, default=60)
    parser.add_argument('--h_background', type=float, default=4000)
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
    add_central_column(h)

    # create initial plot
    x = np.linspace(0, 100, args.ni)
    y = np.linspace(0, 100, args.nj)
    x, y = np.meshgrid(x, y)
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    clear_axes_and_plot_surface(axes, x, y, h)

    # create loop to update plot and progress simulation forwards
    _ = animation.FuncAnimation(figure,
            timestep_and_update_image,
            fargs=(args.dt,
                   lambda: clear_axes_and_plot_surface(axes, x, y, h),
                   lambda: timestep(u, v, h, args)),
                   interval=1)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)