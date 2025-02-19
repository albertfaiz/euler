# /Users/faizahmad/euler/src/euler_solver.py
import numpy as np
from numba import njit
from src.ode_model import ODEModel

@njit
def euler_step(f, y, t, dt):
    """
    Perform one step of Euler's method: y_{n+1} = y_n + dt * f(y_n, t_n).
    """
    return y + dt * f(y, t)

class EulerSolver:
    """
    Solver for ODEs using the Euler method.
    """
    def __init__(self, model: ODEModel) -> None:
        self.model = model

    def solve(self) -> (np.ndarray, np.ndarray):
        """
        Solve the ODE using Euler's method.
        
        Returns:
            t_values (np.ndarray): Array of time points.
            y_values (np.ndarray): Array of solution values at the corresponding times.
        """
        # Use the correct attribute names from ODEModel
        t0 = self.model.t0  
        tf = self.model.tf  
        dt = self.model.dt
        
        t_values = np.arange(t0, tf + dt, dt)
        y_values = np.empty(t_values.shape, dtype=np.float64)
        y_values[0] = self.model.y0
        
        for i in range(len(t_values) - 1):
            y_values[i+1] = euler_step(self.model.f, y_values[i], t_values[i], dt)
            
        return t_values, y_values
