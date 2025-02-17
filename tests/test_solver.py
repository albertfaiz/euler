import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from numba import njit
from ode_model import ODEModel
from euler_solver import EulerSolver

@njit
def decay(y, t):
    return -y

def exact_exponential(y0, t, k=1.0):
    return y0 * np.exp(-k * t)

def test_euler_exponential():
    y0 = 1.0
    t0 = 0.0
    tf = 2.0
    dt = 0.01
    model = ODEModel(f=decay, y_0=y0, t_0=t0, t_f=tf, dt=dt)
    solver = EulerSolver(model)
    t_values, y_values = solver.solve()
    exact = exact_exponential(y0, tf)
    assert abs(y_values[-1] - exact) < 1e-2, "Euler solver error exceeds tolerance."

if __name__ == '__main__':
    test_euler_exponential()
    print("Test passed: Euler solver matches the exponential decay exact solution.")
