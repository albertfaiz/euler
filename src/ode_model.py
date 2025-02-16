# src/ode_model.py
from typing import Callable

class ODEModel:
    """
    Encapsulates an ODE of the form y'(t) = f(y, t) along with its parameters.
    """
    def __init__(self, 
                 f: Callable[[float, float], float], 
                 y_0: float = 1.0, 
                 t_0: float = 0.0, 
                 t_f: float = 1.0, 
                 dt: float = 0.1):
        if t_f <= t_0:
            raise ValueError("Final time t_f must be greater than start time t_0.")
        self.f = f
        self.y_0 = y_0
        self.t_0 = t_0
        self.t_f = t_f
        self.dt = dt

    def call(self, y: float, t: float) -> float:
        """
        Evaluate the ODE function f at state y and time t.
        """
        return self.f(y, t)

# Example usage:
def decay(y: float, t: float = None) -> float:
    """Exponential decay: y'(t) = -y"""
    return -y

if __name__ == '__main__':
    ode = ODEModel(f=decay, y_0=1.0, t_0=0.0, t_f=1.0, dt=0.1)
    print(ode.call(1.0, 0.0))  # Expected: -1.0
