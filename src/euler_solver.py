# src/euler_solver.py
import numpy as np
from numba import njit

@njit
def euler_step(f, y, t, dt):
    """
    Perform one step of Euler's method: y_{n+1} = y_n + dt * f(y_n, t_n)
    """
    return y + dt * f(y, t)

class EulerSolver:
    """
    Euler solver for solving an ODE using the Euler method.
    """
    def __init__(self, model):
        self.model = model

    def solve(self):
        t0 = self.model.t_0
        tf = self.model.t_f
        dt = self.model.dt
        num_steps = int((tf - t0) / dt) + 1
        t_values = np.linspace(t0, tf, num_steps)
        y_values = np.empty(num_steps)
        y_values[0] = self.model.y_0
        # Use Euler's method via the jitted euler_step
        for n in range(num_steps - 1):
            y_values[n+1] = euler_step(self.model.f, y_values[n], t_values[n], dt)
        return t_values, y_values

if __name__ == '__main__':
    from ode_model import ODEModel
    from numba import njit

    @njit
    def decay(y, t):
        return -y

    # Create an ODE model with the jitted decay function
    model = ODEModel(f=decay, y_0=1.0, t_0=0.0, t_f=2.0, dt=0.01)
    solver = EulerSolver(model)
    t, y = solver.solve()
    print("Final numerical solution:", y[-1])
