"""
Orbital-simulations — Barnes-Hut N-body gravitational simulation.

Quick start
-----------
>>> from barnes_hut import Point, Simulation, make_exponential_disk
>>> particles = make_exponential_disk(n=500, G=0.1)
>>> sim = Simulation(particles, G=0.1, theta=0.6, eps=0.1, dt=0.05)
>>> sim.run(n_steps=300, show=True)

See ``barnes_hut`` module for full API.
"""

from .barnes_hut import *  # noqa: F401,F403
from .barnes_hut import __version__
