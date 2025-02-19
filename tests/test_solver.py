import numpy as np
import pytest
from src.ode_model import ODEModel
from src.euler_solver import EulerSolver

def decay_ode(y: float, t: float) -> float:
    return -y

def logistic_ode(y: float, t: float, r: float = 1.0, K: float = 10.0) -> float:
    return r * y * (1 - y / K)

def exact_decay(y0: float, t: float, k: float = 1.0) -> float:
    return y0 * np.exp(-k * t)

def exact_logistic(y0: float, t: float, r: float = 1.0, K: float = 10.0) -> float:
    return K / (1 + ((K - y0) / y0) * np.exp(-r * t))

def test_euler_exponential():
    y0, t0, tf, dt = 1.0, 0.0, 1.0, 0.01
    model = ODEModel(f=decay_ode, y0=y0, t0=t0, tf=tf, dt=dt)
    solver = EulerSolver(model)
    t_values, y_values = solver.solve()
    
    y_exact = exact_decay(y0, tf)
    y_numeric = y_values[-1]
    tol = 1e-2
    assert abs(y_numeric - y_exact) < tol, f"Numerical: {y_numeric}, Exact: {y_exact}"

def test_euler_logistic():
    y0, t0, tf, dt = 1.0, 0.0, 5.0, 0.01
    r, K = 1.0, 10.0
    # Use a lambda function to capture logistic parameters
    ode_func = lambda y, t: logistic_ode(y, t, r, K)
    model = ODEModel(f=ode_func, y0=y0, t0=t0, tf=tf, dt=dt)
    solver = EulerSolver(model)
    t_values, y_values = solver.solve()
    
    y_exact = exact_logistic(y0, tf, r, K)
    y_numeric = y_values[-1]
    tol = 0.5
    assert abs(y_numeric - y_exact) < tol, f"Numerical: {y_numeric}, Exact: {y_exact}"
