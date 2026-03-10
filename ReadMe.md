# Barnes-Hut N-Body Simulation

A Python library for simulating self-gravitating particles in 2D using the **Barnes-Hut algorithm** — an O(N log N) approximation to the O(N²) direct pairwise gravity problem.

![Galaxy simulation](https://github.com/Astrojigs/Orbital-simulations/blob/main/Outputs/GIF/Barnes_hut_dual_gif.gif)

## Features

- **Barnes-Hut quadtree** with configurable opening angle (`theta`), Plummer softening, and max-depth guard
- **Leapfrog integrator** (kick-drift-kick) for long-term energy conservation
- **Initial-condition generators**: exponential disk (galaxy), Plummer sphere (star cluster), uniform random, solar system
- **Diagnostics**: kinetic/potential/total energy, momentum, center of mass
- **Visualization**: quadtree structure, particle scatter, optional MP4 video export
- **Direct O(N²) solver** included for comparison and small-N use

## Quick Start

```python
from barnes_hut import Simulation, make_exponential_disk

# Generate a 500-particle galaxy
particles = make_exponential_disk(n=500, G=0.1)

# Create and run a simulation
sim = Simulation(particles, G=0.1, theta=0.6, eps=0.15, dt=0.06)
sim.run(n_steps=300, show=True, viewport=(-45, 45, -45, 45))
```

## Installation

Clone the repository and import directly:

```bash
git clone https://github.com/Astrojigs/Orbital-simulations.git
cd Orbital-simulations
```

```python
from barnes_hut import *
```

**Dependencies**: `numpy`, `matplotlib` (optional: `opencv-python` for video export)

## API Overview

### Data Structures

| Class | Description |
|-------|-------------|
| `Point(x, y, mass, vx, vy)` | A 2D particle with position, velocity, and mass |
| `Rectangle(cx, cy, w, h)` | Axis-aligned bounding box |
| `Quadtree(boundary, theta, capacity, eps)` | Barnes-Hut quadtree node |

### Simulation

| Class/Function | Description |
|----------------|-------------|
| `Simulation(particles, G, theta, eps, dt)` | Leapfrog integrator with Barnes-Hut forces |
| `sim.step()` | Advance one time-step |
| `sim.run(n_steps, show, save_video, ...)` | Run multiple steps with optional rendering |
| `build_tree(particles, theta, ...)` | Build a quadtree from a list of particles |

### Initial Conditions

| Function | Description |
|----------|-------------|
| `make_exponential_disk(n, G, ...)` | Galaxy with exponential disk + Gaussian bulge |
| `make_plummer_sphere(n, M_total, a, ...)` | Isotropic star cluster (Plummer model) |
| `make_random_uniform(n, width, ...)` | Uniformly distributed particles |
| `make_solar_system()` | Sun + 8 planets with real masses/velocities |

### Diagnostics

| Function | Description |
|----------|-------------|
| `kinetic_energy(points)` | Total KE |
| `potential_energy(points, G, eps)` | Pairwise PE (O(N²)) |
| `total_energy(points, G, eps)` | KE + PE |
| `total_momentum(points)` | Total linear momentum (px, py) |
| `center_of_mass(points)` | Mass-weighted center position |
| `direct_nbody_step(points, dt, G, eps)` | One O(N²) leapfrog step for comparison |

## Repository Structure

```
├── barnes_hut.py                             # Core library (all classes and functions)
├── __init__.py                               # Package exports
├── Examples/
│   └── Using_barnes_hut.ipynb                # Comprehensive tutorial and examples
├── Barnes-hut Algorithm Animations.ipynb     # Galaxy simulations at various scales
├── Original Gravity Animation.ipynb          # Direct N-body demos (planets, solar system)
├── Gravity Animation with potential contour.ipynb  # Potential field visualization
├── Outputs/                                  # Generated animations and images
├── LICENSE.md                                # MIT License
└── ReadMe.md                                 # This file
```

## How It Works

The Barnes-Hut algorithm replaces the O(N²) pairwise force calculation with a hierarchical approximation:

1. **Build a quadtree** — recursively subdivide space so each leaf contains at most one particle
2. **Walk the tree for each particle** — if a tree node is "far enough" away (cell size / distance < theta), treat the entire node as a single body at its center of mass
3. **Integrate** — use a symplectic leapfrog scheme to update positions and velocities

The `theta` parameter controls the accuracy/speed tradeoff: `theta = 0` gives exact O(N²) forces; `theta = 1` is very approximate but fast.

## Examples

See the [Examples notebook](Examples/Using_barnes_hut.ipynb) for:

- Quadtree construction and visualization
- Force accuracy comparison (Barnes-Hut vs direct)
- Galaxy and star-cluster simulations
- Performance benchmarking (N vs wall time)
- Energy conservation verification

## Contributing

Contributions welcome! Submit bug reports, feature requests, or pull requests through GitHub.

## License

MIT License. See [LICENSE.md](LICENSE.md).

## Acknowledgments

Thanks to [@iamstarstuff](https://github.com/iamstarstuff) for support throughout development.
