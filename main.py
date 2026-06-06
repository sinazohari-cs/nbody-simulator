from visualizer import run_simulation
from presets import figure_eight

bodies = figure_eight()
run_simulation(
    initial_bodies=bodies,
    xlim=(-2, 2),
    ylim=(-2, 2),
    use_normalized=True,
    dt=0.0005
)
