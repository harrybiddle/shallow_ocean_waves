import math
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveTimestepper():

    def __init__(self, u, v, h, timestep, dt, t=0, epsilon=1,
                 max_steps=1000):
        ''' Implements the simple Euler 2-Step Adaptive Step Size algorithm
        from http://www.math.ubc.ca/~feldman/math/vble.pdf.
        '''
        # ingest all arguments to self
        for name, value in vars().items():
            if name != 'self':
                setattr(self, name, value)

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
            np.copyto(src=self.u, dst=self.u1)
            np.copyto(src=self.v, dst=self.v1)
            np.copyto(src=self.h, dst=self.h1)

            np.copyto(src=self.u, dst=self.u2)
            np.copyto(src=self.v, dst=self.v2)
            np.copyto(src=self.h, dst=self.h2)

            # take one step forwards by dt
            self.timestep(self.u1, self.v1, self.h1, dt)

            # take two half-steps forwards by dt/2
            self.timestep(self.u2, self.v2, self.h2, dt / 2)
            self.timestep(self.u2, self.v2, self.h2, dt / 2)

            # compare the two in order to generate an error estimate
            # only comparing height field here: could go further and compare u
            # and v too, but this seems to be OK
            E = np.linalg.norm(self.h2 - self.h1)
            r = E / dt

            # break out if the error is accceptible
            error_below_threshold = (r < self.epsilon)
            if error_below_threshold:
                self.dt = dt
                self.t += dt
                # combine the two solutions to get the lowest error possible
                # TODO replace with a swap for speed
                np.copyto(src=self.u, dst=(2 * self.u2 - self.u1))
                np.copyto(src=self.v, dst=(2 * self.v2 - self.v1))
                np.copyto(src=self.h, dst=(2 * self.h2 - self.h1))
                return steps, self.dt

            # repeat with a reduced dt if the error is too high
            dt = 0.9 * self.epsilon * dt / r

    def time(self):
        return self.t
