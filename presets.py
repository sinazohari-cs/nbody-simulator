import numpy as np
from bodies import Body

def figure_eight():
    p1 = np.array([ 0.97000436, -0.24308753])
    p2 = np.array([-0.97000436,  0.24308753])
    p3 = np.array([ 0.0,         0.0       ])
    v3 = np.array([ 0.93240737,  0.86473146])
    v1 = -v3 / 2; v2 = -v3 / 2
    return [
        Body(mass=1.0, position=p1.copy(), velocity=v1.copy(), color='#E24B4A', name='Alpha'),
        Body(mass=1.0, position=p2.copy(), velocity=v2.copy(), color='#378ADD', name='Beta'),
        Body(mass=1.0, position=p3.copy(), velocity=v3.copy(), color='#1D9E75', name='Gamma'),
    ]

def binary_star():
    G = 6.674e-11
    mass = 1e30
    dist = 1.5e11
    v = np.sqrt(G * mass / (2 * dist))
    return [
        Body(mass=mass, position=np.array([0.0, 0.0]),   velocity=np.array([0.0,  v]), color='#EF9F27', name='Castor A'),
        Body(mass=mass, position=np.array([dist, 0.0]),  velocity=np.array([0.0, -v]), color='#FAC775', name='Castor B'),
    ]

def solar_system():
    G = 6.674e-11
    M = 1.989e30
    bodies = [Body(mass=M, position=np.array([0.0,0.0]), velocity=np.array([0.0,0.0]), color='#EF9F27', name='Sun')]
    planets = [
        ('Mercury', 3.285e23, 5.791e10, '#B4B2A9'),
        ('Venus',   4.867e24, 1.082e11, '#FAC775'),
        ('Earth',   5.972e24, 1.496e11, '#378ADD'),
        ('Mars',    6.390e23, 2.279e11, '#E24B4A'),
        ('Jupiter', 1.898e27, 7.783e11, '#D85A30'),
    ]
    for name, mass, dist, color in planets:
        v = np.sqrt(G * M / dist)
        bodies.append(Body(mass=mass, position=np.array([dist,0.0]),
                           velocity=np.array([0.0, v]), color=color, name=name))
    return bodies

def chaos_three():
    G = 6.674e-11
    mass = 1e30
    dist = 1.5e11
    v = np.sqrt(G * mass / (2 * dist))
    return [
        Body(mass=mass,   position=np.array([0.0, 0.0]),        velocity=np.array([0.0,  v]),   color='#E24B4A', name='Chaos A'),
        Body(mass=mass,   position=np.array([dist, 0.0]),       velocity=np.array([0.0, -v]),   color='#378ADD', name='Chaos B'),
        Body(mass=mass*2, position=np.array([0.75e11, 1.3e11]), velocity=np.array([v*0.3, 0.0]),color='#1D9E75', name='Chaos C'),
    ]

def figure_eight_normalized():
    return figure_eight()

def lagrange_points():
    G = 6.674e-11
    M1 = 2e30
    M2 = 1e29
    dist = 2e11
    v1 = np.sqrt(G * M2**2 / ((M1+M2)*dist))
    v2 = np.sqrt(G * M1**2 / ((M1+M2)*dist))
    # L4 trojan
    angle = np.pi/3
    r_trojan = dist
    pos_trojan = np.array([r_trojan*np.cos(angle), r_trojan*np.sin(angle)])
    v_trojan = np.sqrt(G*(M1+M2)/dist)
    v_trojan_vec = np.array([-np.sin(angle), np.cos(angle)]) * v_trojan * M2/(M1+M2)
    return [
        Body(mass=M1, position=np.array([0.0,0.0]),   velocity=np.array([0.0, v1]),  color='#EF9F27', name='Star'),
        Body(mass=M2, position=np.array([dist,0.0]),  velocity=np.array([0.0,-v2]),  color='#378ADD', name='Planet'),
        Body(mass=1e24, position=pos_trojan, velocity=v_trojan_vec, color='#1D9E75', name='Trojan'),
    ]