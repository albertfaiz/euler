# src/euler_solver.py
import numpy as np
from numba import njit
from ode_model import ODEModel  # Import the ODEModel class

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
    def __init__(self, model: ODEModel):
        self.model = model

    def solve(self):
        t0 = self.model.t_0
        tf = self.model.t_f
        dt = self.model.dt
        num_steps = int((tf - t0) / dt) + 1
        t_values = np.linspace(t0, tf, num_steps)
        y_values = np.empty(num_steps)
        y_values[0] = self.model.y_0
        # Iteratively compute the solution using Euler's method
        for n in range(num_steps - 1):
            y_values[n+1] = euler_step(self.model.f, y_values[n], t_values[n], dt)
        return t_values, y_values

# Example usage:
if __name__ == '__main__':
    from ode_model import ODEModel

    def decay(y: float, t: float) -> float:
        return -y

    model = ODEModel(f=decay, y_0=1.0, t_0=0.0, t_f=5.0, dt=0.1)
    solver = EulerSolver(model)
    t, y = solver.solve()
    print("Final numerical solution:", y[-1])
