import numpy as np

G = 6.674e-11
SOFTENING = 1e8

def gravitational_force(a, b):
    r_vec = b.position - a.position
    r2 = np.dot(r_vec, r_vec) + SOFTENING**2
    r = np.sqrt(r2)
    if r < 1e6:
        return np.zeros(2)
    force = (G * a.mass * b.mass / r2) * (r_vec / r)
    return force

def compute_accelerations(bodies):
    accelerations = []
    for body in bodies:
        total_force = np.zeros(2)
        for other in bodies:
            if other is not body:
                total_force += gravitational_force(body, other)
        accelerations.append(total_force / body.mass)
    return accelerations

def rk4_step(bodies, dt):
    positions  = [b.position.copy() for b in bodies]
    velocities = [b.velocity.copy() for b in bodies]

    acc = compute_accelerations(bodies)
    k1_v = [b.velocity.copy() for b in bodies]; k1_a = acc

    for i, b in enumerate(bodies):
        b.position = positions[i] + k1_v[i] * dt/2
        b.velocity = velocities[i] + k1_a[i] * dt/2
    acc = compute_accelerations(bodies)
    k2_v = [b.velocity.copy() for b in bodies]; k2_a = acc

    for i, b in enumerate(bodies):
        b.position = positions[i] + k2_v[i] * dt/2
        b.velocity = velocities[i] + k2_a[i] * dt/2
    acc = compute_accelerations(bodies)
    k3_v = [b.velocity.copy() for b in bodies]; k3_a = acc

    for i, b in enumerate(bodies):
        b.position = positions[i] + k3_v[i] * dt
        b.velocity = velocities[i] + k3_a[i] * dt
    acc = compute_accelerations(bodies)
    k4_v = [b.velocity.copy() for b in bodies]; k4_a = acc

    for i, b in enumerate(bodies):
        b.position = positions[i] + (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i]) * dt/6
        b.velocity = velocities[i] + (k1_a[i] + 2*k2_a[i] + 2*k3_a[i] + k4_a[i]) * dt/6

def total_energy(bodies):
    KE = sum(0.5 * b.mass * np.dot(b.velocity, b.velocity) for b in bodies)
    PE = 0.0
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            r = np.linalg.norm(bodies[j].position - bodies[i].position)
            if r > 1e6:
                PE -= G * bodies[i].mass * bodies[j].mass / r
    return KE + PE

def center_of_mass(bodies):
    total_mass = sum(b.mass for b in bodies)
    if total_mass == 0:
        return np.zeros(2)
    return sum(b.mass * b.position for b in bodies) / total_mass

def total_momentum(bodies):
    return sum(b.mass * b.velocity for b in bodies)