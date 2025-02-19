# /Users/faizahmad/euler/src/ode_model.py
from typing import Callable

class ODEModel:
    """
    Encapsulates an ODE of the form y'(t) = f(y, t) along with its parameters.
    """
    def __init__(self, 
                 f: Callable[[float, float], float],
                 y0: float = 1.0, 
                 t0: float = 0.0, 
                 tf: float = 1.0, 
                 dt: float = 0.1) -> None:
        if tf <= t0:
            raise ValueError("Final time tf must be greater than start time t0.")
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
            
        self.f = f         # The ODE function f(y, t)
        self.y0 = y0       # Initial condition
        self.t0 = t0       # Start time
        self.tf = tf       # End time
        self.dt = dt       # Time step

    def call(self, y: float, t: float) -> float:
        """
        Evaluate the ODE function at the given state y and time t.
        """
        return self.f(y, t)
