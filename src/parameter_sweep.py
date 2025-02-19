import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from src.ode_model import ODEModel
from src.euler_solver import EulerSolver

def decay_ode(y: float, t: float) -> float:
    return -y

def exact_decay(y0: float, t: float, k: float = 1.0) -> float:
    return y0 * np.exp(-k * t)

def run_one_case(params):
    """
    Run one simulation with a given dt and return the final error and runtime.
    
    Args:
        params (tuple): (index, dt)
        
    Returns:
        tuple: (index, dt, final_error, runtime)
    """
    i, dt = params
    y0, t0, tf = 1.0, 0.0, 1.0
    model = ODEModel(f=decay_ode, y0=y0, t0=t0, tf=tf, dt=dt)
    solver = EulerSolver(model)
    
    start_time = time.time()
    t_values, y_values = solver.solve()
    runtime = time.time() - start_time
    
    y_numeric = y_values[-1]
    y_exact_val = exact_decay(y0, tf)
    final_error = abs(y_numeric - y_exact_val)
    
    return (i, dt, final_error, runtime)

if __name__ == '__main__':
    m = 8
    dt_values = [1/(2**i) for i in range(m+1)]
    params_list = [(i, dt) for i, dt in enumerate(dt_values)]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_one_case, params_list))
    
    # Save results to a file for later analysis
    with open("results.txt", "w") as f:
        f.write("index,dt,final_error,runtime\n")
        for res in results:
            f.write(f"{res[0]},{res[1]},{res[2]},{res[3]}\n")
    
    print("Parameter sweep completed. Results saved to results.txt")
