# N-Body Gravitational Simulator

A real-time gravitational simulator built from scratch in Python.
Uses **RK4 (Runge-Kutta 4th order)** numerical integration and **Newtonian gravity**.

## Run

```bash
uv run main.py
```

## Controls

| Action | Effect |
|--------|--------|
| Click + drag on canvas | Launch a new body (drag direction = velocity) |
| Space | Pause / Resume |
| Scroll / toolbar zoom | Zoom in/out |

## Features

- RK4 integrator for stable, accurate orbits
- Fading trails showing orbital history
- Real-time energy conservation graph (% drift from initial)
- Live stats: simulated time, total energy, momentum, center of mass
- Velocity vector overlay toggle
- Follow center of mass mode
- Grid toggle
- 5 built-in presets: Figure-8, Binary Star, Solar System, Chaos 3-Body, Lagrange/Trojan
- Mass slider for new bodies
- Softening parameter prevents singularities on close approach

## Presets

| Preset | Description |
|--------|-------------|
| Figure-8 Orbit | Famous periodic 3-body solution (Chenciner & Montgomery 2000) |
| Binary Star | Two equal-mass stars in circular mutual orbit |
| Solar System | Sun + Mercury, Venus, Earth, Mars, Jupiter with real masses |
| Chaos 3-Body | Unstable 3-body — watch for ejections |
| Lagrange / Trojan | Star + planet + Trojan asteroid at L4 point |

## Architecture

```
bodies.py     — Body class (mass, position, velocity, acceleration)
physics.py    — Gravitational force, RK4 integrator, energy, momentum
presets.py    — Initial condition configurations
visualizer.py — Matplotlib animation, UI, interaction
main.py       — Entry point
```
