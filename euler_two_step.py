import math
import matplotlib.pyplot as plt

class AdaptiveTimestepper():

    def __init__(self, y0, f, starting_h, t0=0, epsilon=1):
        self.y = y0
        self.f = f
        self.h = starting_h
        self.t = t0
        self.epsilon = epsilon
        self.max_steps = 1000

    def step_forwards(self):
        steps = 0
        h, y, t = self.h, self.y, self.t
        while True:
            # bail out if we're not converging
            if steps > self.max_steps:
                raise RuntimeError('Not converged after {} steps'.format(steps))
            steps += 1

            # take one step forwards by h
            A1 = y + h * self.f(t, y)

            # take two half-steps forwards by h/2
            A2 =  y + h/2 * self.f(t,       y)
            A2 = A2 + h/2 * self.f(t + h/2, A2)

            # compare the two in order to generate an error estimate
            E = (A1 - A2)
            r = math.fabs(E) / h

            # break out if the error is accceptible
            error_below_threshold = (r < self.epsilon)
            if error_below_threshold:
                self.h = h
                self.t += h
                self.y = 2 * A2 - A1
                return steps, h, self.y

            # repeat with a reduced h if the error is too high
            h = 0.9 * self.epsilon * h / r

    def time(self):
        return self.t

def main():

    # problem definition
    f0 = math.exp(-2)
    t_final = 10
    def f(t, y):
        return 8 * (1 - 2 * t) * y

    # analytical solution
    def f_solution(t):
        return f0 * math.exp(-8 * (t - 1) * t)

    def solve_numerically(epsilon, starting_h):
        # set up a timestepper
        timestepper = AdaptiveTimestepper(f0, f, starting_h, epsilon=epsilon)
        ts = [0]
        ys = [f0]
        while timestepper.time() < t_final:
            _, _, solution = timestepper.step_forwards()
            ts.append(timestepper.time())
            ys.append(solution)

        # create the errors from the analytical solution
        errors = []
        for t, y in zip(ts, ys):
            analytical_solution = f_solution(t)
            error = math.fabs(y - analytical_solution)
            errors.append(error)

        return ts, errors

    def plot(epsilon, starting_h):
        ts, errors = solve_numerically(epsilon, starting_h)
        return plt.plot(ts, errors, label=str(epsilon))[0]

    # generate lines
    handles = []
    handles.append(plot(epsilon=0.1, starting_h=1))
    handles.append(plot(epsilon=0.05, starting_h=1))
    handles.append(plot(epsilon=0.01, starting_h=1))

    # create and show final plot
    plt.legend(handles=handles)
    plt.show()

if __name__ == '__main__':
    main()