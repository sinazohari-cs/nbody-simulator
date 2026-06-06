import numpy as np

class Body:
    def __init__(self, mass, position=None, velocity=None, color=None, name=None, radius=None):
        self.mass = mass
        self.position = position if position is not None else np.zeros(2)
        self.velocity = velocity if velocity is not None else np.zeros(2)
        self.acceleration = np.zeros(2)
        self.color = color
        self.name = name
        self.radius = radius  # optional display radius override

    def __repr__(self):
        return f"Body(mass={self.mass:.2e}, pos={self.position}, vel={self.velocity})"