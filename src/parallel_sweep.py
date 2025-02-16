# src/parallel_sweep.py
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ode_model import ODEModel
from euler_solver import EulerSolver

def exact_exponential(y0, t, k=1.0):
    return y0 * np.exp(-k * t)

def run_one_case(dt):
    def decay(y, t):
        return -y
    y0 = 1.0
    t0 = 0.0
    tf = 2.0
    model = ODEModel(f=decay, y_0=y0, t_0=t0, t_f=tf, dt=dt)
    solver = EulerSolver(model)
    start = time.time()
    t_values, y_values = solver.solve()
    runtime = time.time() - start
    exact = exact_exponential(y0, tf)
    error = abs(y_values[-1] - exact)
    return dt, error, runtime

if __name__ == '__main__':
    dt_values = [1.0 / (2 ** i) for i in range(5)]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_one_case, dt_values))
    for dt, error, runtime in results:
        print(f"dt: {dt}, error: {error}, runtime: {runtime}")
