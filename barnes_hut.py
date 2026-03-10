"""
barnes_hut — Barnes-Hut N-body gravitational simulation in 2D.

A self-contained library for simulating self-gravitating particles using:

- **Quadtree** spatial decomposition with the Barnes-Hut opening criterion
  (s/d < theta) for O(N log N) force approximation.
- **Leapfrog** (kick-drift-kick) symplectic integrator for long-term
  energy conservation.
- **Plummer softening** to regularize close encounters.
- Built-in initial-condition generators (exponential disk, Plummer sphere,
  uniform random, solar system).
- Built-in energy/momentum diagnostics.
- Visualization helpers for particles and quadtree structure.

Typical usage
-------------
>>> from barnes_hut import Point, Simulation, make_exponential_disk
>>> particles = make_exponential_disk(n=500, G=0.1)
>>> sim = Simulation(particles, G=0.1, theta=0.6, eps=0.1, dt=0.05)
>>> sim.run(n_steps=300, show=True)

Author: Jigar Patel
License: MIT
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import numpy as np

# Optional imports — guarded so the core library works without them.
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False

try:
    from IPython.display import clear_output

    _HAS_IPYTHON = True
except ImportError:  # pragma: no cover
    _HAS_IPYTHON = False

try:
    import cv2  # type: ignore[import-untyped]

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


__all__ = [
    # Data structures
    "Point",
    "Rectangle",
    "Quadtree",
    # Simulation
    "Simulation",
    # Initial-condition generators
    "make_exponential_disk",
    "make_plummer_sphere",
    "make_random_uniform",
    "make_solar_system",
    # Diagnostics
    "kinetic_energy",
    "potential_energy",
    "total_energy",
    "total_momentum",
    "center_of_mass",
    # Direct N-body (for comparison / small-N use)
    "direct_nbody_step",
]

__version__ = "2.0.0"


# =====================================================================
#  Data structures
# =====================================================================


@dataclass
class Point:
    """A 2-D particle with position, velocity, and mass.

    Parameters
    ----------
    x, y : float
        Cartesian position.
    mass : float
        Particle mass (must be > 0).
    vx, vy : float
        Velocity components.
    """

    x: float
    y: float
    mass: float = 1.0
    vx: float = 0.0
    vy: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError(f"mass must be positive, got {self.mass}")
        for attr in ("x", "y", "vx", "vy", "mass"):
            v = getattr(self, attr)
            if not math.isfinite(v):
                raise ValueError(f"{attr} must be finite, got {v}")

    def speed(self) -> float:
        """Return scalar speed."""
        return math.hypot(self.vx, self.vy)

    def distance_to(self, other: "Point") -> float:
        """Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return (
            f"Point(x={self.x:.4g}, y={self.y:.4g}, mass={self.mass:.4g}, "
            f"vx={self.vx:.4g}, vy={self.vy:.4g})"
        )


class Rectangle:
    """Axis-aligned bounding box defined by center ``(cx, cy)`` and
    full width/height ``(w, h)``.

    Containment is *inclusive* on all edges to avoid cracks between
    adjacent quadtree cells.
    """

    __slots__ = ("cx", "cy", "w", "h", "west", "east", "south", "north")

    def __init__(self, cx: float, cy: float, w: float, h: float) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"Rectangle dimensions must be positive, got w={w}, h={h}")
        self.cx = float(cx)
        self.cy = float(cy)
        self.w = float(w)
        self.h = float(h)
        self.west = self.cx - self.w / 2
        self.east = self.cx + self.w / 2
        self.south = self.cy - self.h / 2
        self.north = self.cy + self.h / 2

    # Keep backward-compatible attribute names
    @property
    def x(self) -> float:
        return self.cx

    @property
    def y(self) -> float:
        return self.cy

    def contains(self, p: Point) -> bool:
        """Return True if ``p`` lies inside (inclusive edges)."""
        return (self.west <= p.x <= self.east) and (self.south <= p.y <= self.north)

    def area(self) -> float:
        return self.w * self.h

    def draw(self, ax: "Axes", color: str = "red", lw: float = 0.6, **kwargs) -> None:
        """Draw the rectangle outline on a Matplotlib axes."""
        _require_matplotlib()
        xs = [self.west, self.east, self.east, self.west, self.west]
        ys = [self.north, self.north, self.south, self.south, self.north]
        ax.plot(xs, ys, c=color, lw=lw, **kwargs)

    def __repr__(self) -> str:
        return f"Rectangle(cx={self.cx:.4g}, cy={self.cy:.4g}, w={self.w:.4g}, h={self.h:.4g})"


# =====================================================================
#  Quadtree
# =====================================================================

# Quadrant index convention:
#   0 = NW (top-left)    1 = NE (top-right)
#   2 = SW (bottom-left)  3 = SE (bottom-right)
_QUAD_NAMES = ("NW", "NE", "SW", "SE")


class Quadtree:
    """Barnes-Hut quadtree for 2-D gravitational force approximation.

    Parameters
    ----------
    boundary : Rectangle
        Spatial extent of this node.
    theta : float
        Opening-angle parameter.  Typical range 0.3-1.0.
        Smaller = more accurate but slower.
    capacity : int
        Maximum particles stored in a leaf before subdivision.
    eps : float
        Plummer softening length (adds ``eps**2`` to ``r**2``).
    max_depth : int
        Safety cap on recursion depth.

    Attributes
    ----------
    mass : float
        Total mass of all particles in this subtree.
    com_x, com_y : float
        Center of mass of this subtree (updated incrementally on insert).
    """

    __slots__ = (
        "boundary",
        "theta",
        "capacity",
        "eps",
        "max_depth",
        "mass",
        "com_x",
        "com_y",
        "points",
        "children",
        "divided",
        "_n_particles",
    )

    def __init__(
        self,
        boundary: Rectangle,
        theta: float = 0.5,
        capacity: int = 1,
        eps: float = 1e-3,
        max_depth: int = 40,
    ) -> None:
        self.boundary = boundary
        self.theta = float(theta)
        self.capacity = max(1, int(capacity))
        self.eps = float(eps)
        self.max_depth = int(max_depth)

        self.mass: float = 0.0
        self.com_x: float = 0.0
        self.com_y: float = 0.0

        self.points: List[Point] = []
        self.children: List[Optional["Quadtree"]] = [None, None, None, None]
        self.divided: bool = False
        self._n_particles: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quadrant_index(self, x: float, y: float) -> int:
        """Map ``(x, y)`` to a child index 0..3.

        Tie-breaking: ``x <= cx`` -> left; ``y > cy`` -> top.
        """
        left = x <= self.boundary.cx
        top = y > self.boundary.cy
        if top and left:
            return 0
        if top and (not left):
            return 1
        if (not top) and left:
            return 2
        return 3

    def _child_bounds(self, qi: int) -> Rectangle:
        """Return the bounding Rectangle for child quadrant *qi*."""
        hw = self.boundary.w * 0.25  # quarter of parent width
        hh = self.boundary.h * 0.25
        cx, cy = self.boundary.cx, self.boundary.cy
        if qi == 0:  # NW
            return Rectangle(cx - hw, cy + hh, self.boundary.w / 2, self.boundary.h / 2)
        if qi == 1:  # NE
            return Rectangle(cx + hw, cy + hh, self.boundary.w / 2, self.boundary.h / 2)
        if qi == 2:  # SW
            return Rectangle(cx - hw, cy - hh, self.boundary.w / 2, self.boundary.h / 2)
        # qi == 3, SE
        return Rectangle(cx + hw, cy - hh, self.boundary.w / 2, self.boundary.h / 2)

    def _ensure_child(self, qi: int) -> "Quadtree":
        if self.children[qi] is None:
            self.children[qi] = Quadtree(
                self._child_bounds(qi),
                theta=self.theta,
                capacity=self.capacity,
                eps=self.eps,
                max_depth=self.max_depth,
            )
        return self.children[qi]  # type: ignore[return-value]

    def _subdivide(self, depth: int) -> None:
        """Promote this leaf to an internal node, pushing stored points
        into children."""
        self.divided = True
        old = self.points
        self.points = []
        for p in old:
            qi = self._quadrant_index(p.x, p.y)
            self._ensure_child(qi).insert(p, depth + 1)

    # ------------------------------------------------------------------
    # Public API — insertion
    # ------------------------------------------------------------------

    def insert(self, p: Point, depth: int = 0) -> bool:
        """Insert particle *p* into the tree.

        Returns True if the particle was accepted (inside boundary),
        False otherwise.  Mass and center-of-mass are updated
        incrementally at every node on the insertion path.
        """
        if not self.boundary.contains(p):
            return False

        # Incremental COM update
        m_old = self.mass
        self.mass = m_old + p.mass
        if self.mass > 0:
            self.com_x = (self.com_x * m_old + p.x * p.mass) / self.mass
            self.com_y = (self.com_y * m_old + p.y * p.mass) / self.mass
        self._n_particles += 1

        # Leaf: store if below capacity or at max depth
        if not self.divided:
            if len(self.points) < self.capacity or depth >= self.max_depth:
                self.points.append(p)
                return True
            # Need to subdivide
            self._subdivide(depth)

        # Route into the correct child
        qi = self._quadrant_index(p.x, p.y)
        return self._ensure_child(qi).insert(p, depth + 1)

    def insert_all(self, points: List[Point]) -> int:
        """Convenience: insert many particles, return count of accepted."""
        return sum(self.insert(p) for p in points)

    # ------------------------------------------------------------------
    # Public API — force calculation
    # ------------------------------------------------------------------

    def compute_force(self, p: Point, G: float = 1.0) -> Tuple[float, float]:
        """Compute the gravitational force on particle *p* from this
        subtree using the Barnes-Hut approximation.

        Parameters
        ----------
        p : Point
            Target particle.
        G : float
            Gravitational constant.

        Returns
        -------
        fx, fy : float
            Force components.
        """
        if self.mass == 0:
            return 0.0, 0.0

        dx = self.com_x - p.x
        dy = self.com_y - p.y
        r2 = dx * dx + dy * dy + self.eps * self.eps

        # Leaf node: direct sum over stored particles
        if not self.divided:
            fx = fy = 0.0
            for q in self.points:
                if q is p:
                    continue
                qdx = q.x - p.x
                qdy = q.y - p.y
                qr2 = qdx * qdx + qdy * qdy + self.eps * self.eps
                inv_r3 = 1.0 / (qr2 * math.sqrt(qr2))
                f = G * p.mass * q.mass * inv_r3
                fx += f * qdx
                fy += f * qdy
            return fx, fy

        # Internal node: Barnes-Hut opening criterion
        d = math.sqrt(r2)
        s = max(self.boundary.w, self.boundary.h)
        if d > 0 and (s / d) < self.theta:
            inv_r3 = 1.0 / (r2 * d)
            f = G * p.mass * self.mass * inv_r3
            return f * dx, f * dy

        # Open the node: recurse into children
        fx = fy = 0.0
        for ch in self.children:
            if ch is not None and ch.mass > 0:
                cfx, cfy = ch.compute_force(p, G)
                fx += cfx
                fy += cfy
        return fx, fy

    def compute_acceleration(self, p: Point, G: float = 1.0) -> Tuple[float, float]:
        """Compute acceleration (force / mass) on particle *p*."""
        fx, fy = self.compute_force(p, G)
        return fx / p.mass, fy / p.mass

    # ------------------------------------------------------------------
    # Public API — queries
    # ------------------------------------------------------------------

    @property
    def n_particles(self) -> int:
        """Total number of particles in this subtree."""
        return self._n_particles

    def center_of_mass(self) -> Tuple[float, float]:
        """Return ``(com_x, com_y)`` of this subtree."""
        return self.com_x, self.com_y

    @property
    def nw(self) -> Optional["Quadtree"]:
        return self.children[0]

    @property
    def ne(self) -> Optional["Quadtree"]:
        return self.children[1]

    @property
    def sw(self) -> Optional["Quadtree"]:
        return self.children[2]

    @property
    def se(self) -> Optional["Quadtree"]:
        return self.children[3]

    def depth(self) -> int:
        """Maximum depth of the tree below this node."""
        if not self.divided:
            return 0
        return 1 + max(
            (ch.depth() if ch is not None else 0) for ch in self.children
        )

    def count_nodes(self) -> int:
        """Total number of nodes (leaves + internal) in this subtree."""
        total = 1
        if self.divided:
            for ch in self.children:
                if ch is not None:
                    total += ch.count_nodes()
        return total

    def all_particles(self) -> List[Point]:
        """Collect every particle stored in this subtree (depth-first)."""
        result: List[Point] = []
        self._collect(result)
        return result

    def _collect(self, acc: List[Point]) -> None:
        acc.extend(self.points)
        if self.divided:
            for ch in self.children:
                if ch is not None:
                    ch._collect(acc)

    # ------------------------------------------------------------------
    # Public API — visualization
    # ------------------------------------------------------------------

    def draw(self, ax: Optional["Axes"] = None, color: str = "red",
             lw: float = 0.5, show_empty: bool = True) -> "Axes":
        """Recursively draw all cell boundaries.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If None, use ``plt.gca()``.
        color : str
            Line color.
        lw : float
            Line width.
        show_empty : bool
            If False, skip cells that contain no particles.

        Returns
        -------
        ax : Axes
        """
        _require_matplotlib()
        if ax is None:
            ax = plt.gca()
        if show_empty or self._n_particles > 0:
            self.boundary.draw(ax, color=color, lw=lw)
        if self.divided:
            for ch in self.children:
                if ch is not None:
                    ch.draw(ax, color=color, lw=lw, show_empty=show_empty)
        return ax

    def draw_com(self, ax: Optional["Axes"] = None, **kwargs) -> "Axes":
        """Plot the center of mass of this node (and optionally children)."""
        _require_matplotlib()
        if ax is None:
            ax = plt.gca()
        if self.mass > 0:
            defaults = dict(marker="x", s=40, color="orange", zorder=5)
            defaults.update(kwargs)
            ax.scatter([self.com_x], [self.com_y], **defaults)
        return ax

    def __repr__(self) -> str:
        return (
            f"Quadtree(n={self._n_particles}, mass={self.mass:.4g}, "
            f"com=({self.com_x:.4g}, {self.com_y:.4g}), "
            f"depth={self.depth()}, nodes={self.count_nodes()})"
        )


# =====================================================================
#  Tree construction helper
# =====================================================================


def build_tree(
    points: List[Point],
    theta: float = 0.5,
    capacity: int = 1,
    eps: float = 1e-3,
    max_depth: int = 40,
    padding: float = 1.05,
) -> Quadtree:
    """Build a Quadtree that encloses all *points*.

    The root boundary is a square padded by *padding* around the
    bounding box of all particles.
    """
    if not points:
        raise ValueError("Cannot build tree from an empty particle list.")
    bounds = _square_bounds(points, padding)
    tree = Quadtree(bounds, theta=theta, capacity=capacity, eps=eps, max_depth=max_depth)
    tree.insert_all(points)
    return tree


def _square_bounds(points: List[Point], pad: float = 1.05) -> Rectangle:
    """Compute a padded, square bounding box around *points*."""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    if half <= 0:
        half = 1.0  # degenerate case: all particles at same position
    half *= pad
    return Rectangle(cx, cy, 2 * half, 2 * half)


# =====================================================================
#  Simulation engine
# =====================================================================


class Simulation:
    """Leapfrog (kick-drift-kick) N-body integrator with Barnes-Hut
    force evaluation.

    Parameters
    ----------
    particles : list[Point]
        Particles to simulate (updated **in-place**).
    G : float
        Gravitational constant.
    theta : float
        Barnes-Hut opening angle.
    eps : float
        Softening length.
    dt : float
        Time-step size.
    capacity : int
        Quadtree leaf capacity.
    max_depth : int
        Maximum tree depth.

    Examples
    --------
    >>> pts = make_random_uniform(100, width=100)
    >>> sim = Simulation(pts, G=0.1, dt=0.5)
    >>> sim.step()               # advance one time-step
    >>> sim.run(100, show=True)  # run 100 more steps with live plot
    """

    def __init__(
        self,
        particles: List[Point],
        G: float = 1.0,
        theta: float = 0.5,
        eps: float = 1e-3,
        dt: float = 1.0,
        capacity: int = 1,
        max_depth: int = 40,
    ) -> None:
        if not particles:
            raise ValueError("Need at least one particle.")
        self.particles = particles
        self.G = float(G)
        self.theta = float(theta)
        self.eps = float(eps)
        self.dt = float(dt)
        self.capacity = int(capacity)
        self.max_depth = int(max_depth)

        self.time: float = 0.0
        self.frame: int = 0
        self.tree: Optional[Quadtree] = None

        # Compute initial accelerations
        self._rebuild_tree()
        self._ax, self._ay = self._accelerations()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        self.tree = build_tree(
            self.particles,
            theta=self.theta,
            capacity=self.capacity,
            eps=self.eps,
            max_depth=self.max_depth,
        )

    def _accelerations(self) -> Tuple[np.ndarray, np.ndarray]:
        N = len(self.particles)
        ax = np.empty(N)
        ay = np.empty(N)
        tree = self.tree
        G = self.G
        for i, p in enumerate(self.particles):
            fx, fy = tree.compute_force(p, G)  # type: ignore[union-attr]
            ax[i] = fx / p.mass
            ay[i] = fy / p.mass
        return ax, ay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one time-step (leapfrog KDK)."""
        dt = self.dt
        pts = self.particles
        ax, ay = self._ax, self._ay

        # Half kick
        for i, p in enumerate(pts):
            p.vx += 0.5 * dt * ax[i]
            p.vy += 0.5 * dt * ay[i]

        # Drift
        for p in pts:
            p.x += dt * p.vx
            p.y += dt * p.vy

        # Rebuild tree, new accelerations
        self._rebuild_tree()
        ax, ay = self._accelerations()

        # Half kick
        for i, p in enumerate(pts):
            p.vx += 0.5 * dt * ax[i]
            p.vy += 0.5 * dt * ay[i]

        self._ax, self._ay = ax, ay
        self.time += dt
        self.frame += 1

    def run(
        self,
        n_steps: int = 100,
        *,
        show: bool = False,
        draw_tree: bool = False,
        figsize: Tuple[float, float] = (8, 8),
        point_size: float = 5,
        point_color: str = "black",
        viewport: Optional[Tuple[float, float, float, float]] = None,
        save_video: Optional[str] = None,
        video_fps: int = 30,
        callback: Optional[Callable[["Simulation", int], None]] = None,
    ) -> None:
        """Run *n_steps* of the simulation.

        Parameters
        ----------
        show : bool
            If True, render each frame live (requires matplotlib).
        draw_tree : bool
            Overlay quadtree grid on the plot.
        figsize : tuple
            Matplotlib figure size.
        point_size, point_color : float, str
            Scatter-plot styling.
        viewport : (xmin, xmax, ymin, ymax) or None
            Fixed camera.  None = auto-fit.
        save_video : str or None
            Path to save an MP4 (requires OpenCV).
        video_fps : int
            Frames per second for video.
        callback : callable or None
            ``callback(sim, step_index)`` called after each step, before
            rendering.  Useful for logging diagnostics.
        """
        writer = None
        if save_video is not None:
            if not _HAS_CV2:
                raise RuntimeError(
                    "OpenCV (cv2) is required for video export. "
                    "Install it with: pip install opencv-python"
                )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_video, fourcc, video_fps, (1000, 1000))

        try:
            for t in range(n_steps):
                self.step()

                if callback is not None:
                    callback(self, t)

                if show or writer is not None:
                    self._render_frame(
                        draw_tree=draw_tree,
                        figsize=figsize,
                        point_size=point_size,
                        point_color=point_color,
                        viewport=viewport,
                        writer=writer,
                        show=show,
                        step_idx=t,
                        total_steps=n_steps,
                    )
        finally:
            if writer is not None:
                writer.release()

    def _render_frame(
        self,
        *,
        draw_tree: bool,
        figsize: Tuple[float, float],
        point_size: float,
        point_color: str,
        viewport: Optional[Tuple[float, float, float, float]],
        writer,
        show: bool,
        step_idx: int,
        total_steps: int,
    ) -> None:
        _require_matplotlib()
        fig, ax = plt.subplots(figsize=figsize)
        xs = [p.x for p in self.particles]
        ys = [p.y for p in self.particles]
        ax.scatter(xs, ys, s=point_size, c=point_color, edgecolors="none")

        if draw_tree and self.tree is not None:
            self.tree.draw(ax)

        if viewport is not None:
            ax.set_xlim(viewport[0], viewport[1])
            ax.set_ylim(viewport[2], viewport[3])
        else:
            dx = max(1e-6, 0.05 * max(1.0, max(xs) - min(xs)))
            dy = max(1e-6, 0.05 * max(1.0, max(ys) - min(ys)))
            ax.set_xlim(min(xs) - dx, max(xs) + dx)
            ax.set_ylim(min(ys) - dy, max(ys) + dy)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        plt.tight_layout()

        if writer is not None:
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if show:
            plt.show()
        plt.close(fig)
        if show and _HAS_IPYTHON:
            clear_output(wait=True)

    # ------------------------------------------------------------------
    # Diagnostics (forwarded convenience)
    # ------------------------------------------------------------------

    def kinetic_energy(self) -> float:
        return kinetic_energy(self.particles)

    def potential_energy(self) -> float:
        return potential_energy(self.particles, self.G, self.eps)

    def total_energy(self) -> float:
        return total_energy(self.particles, self.G, self.eps)

    def total_momentum(self) -> Tuple[float, float]:
        return total_momentum(self.particles)

    def center_of_mass(self) -> Tuple[float, float]:
        return center_of_mass(self.particles)


# =====================================================================
#  Diagnostics (standalone functions)
# =====================================================================


def kinetic_energy(points: List[Point]) -> float:
    """Total kinetic energy: sum of 0.5 * m * v**2."""
    return 0.5 * sum(p.mass * (p.vx * p.vx + p.vy * p.vy) for p in points)


def potential_energy(
    points: List[Point], G: float = 1.0, eps: float = 1e-3
) -> float:
    """Pairwise gravitational potential energy (O(N**2) — use for diagnostics)."""
    U = 0.0
    N = len(points)
    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            pj = points[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r = math.sqrt(dx * dx + dy * dy + eps * eps)
            U -= G * pi.mass * pj.mass / r
    return U


def total_energy(
    points: List[Point], G: float = 1.0, eps: float = 1e-3
) -> float:
    """KE + PE (should be roughly conserved in a good integrator)."""
    return kinetic_energy(points) + potential_energy(points, G, eps)


def total_momentum(points: List[Point]) -> Tuple[float, float]:
    """Total linear momentum ``(px, py)``."""
    px = sum(p.mass * p.vx for p in points)
    py = sum(p.mass * p.vy for p in points)
    return px, py


def center_of_mass(points: List[Point]) -> Tuple[float, float]:
    """Mass-weighted center of mass ``(x, y)``."""
    if not points:
        return 0.0, 0.0
    M = sum(p.mass for p in points)
    if M == 0:
        return 0.0, 0.0
    cx = sum(p.mass * p.x for p in points) / M
    cy = sum(p.mass * p.y for p in points) / M
    return cx, cy


# =====================================================================
#  Direct (brute-force) N-body step — for comparison and small N
# =====================================================================


def direct_nbody_step(
    points: List[Point], dt: float = 0.1, G: float = 1.0, eps: float = 1e-3
) -> None:
    """One leapfrog step using exact O(N**2) pairwise forces.

    Useful as a reference for validating the Barnes-Hut approximation
    or for very small N where the tree overhead is not worthwhile.
    Particles are updated **in-place**.
    """
    N = len(points)
    ax = np.zeros(N)
    ay = np.zeros(N)
    eps2 = eps * eps

    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            pj = points[j]
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            r2 = dx * dx + dy * dy + eps2
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            ai = G * pj.mass * inv_r3
            aj = G * pi.mass * inv_r3
            ax[i] += ai * dx
            ay[i] += ai * dy
            ax[j] -= aj * dx
            ay[j] -= aj * dy

    # Leapfrog KDK (half-kick, drift, half-kick in one combined step
    # for a standalone stepper)
    for i, p in enumerate(points):
        p.vx += dt * ax[i]
        p.vy += dt * ay[i]
        p.x += dt * p.vx
        p.y += dt * p.vy


# =====================================================================
#  Initial-condition generators
# =====================================================================


def make_exponential_disk(
    n: int = 400,
    R_d: float = 8.0,
    R_max: float = 40.0,
    M_total: float = 400.0,
    G: float = 0.1,
    bulge_frac: float = 0.15,
    bulge_sigma: float = 2.0,
    rotation: int = 1,
    v_dispersion: float = 0.05,
    center: Tuple[float, float] = (0.0, 0.0),
    seed: int = 42,
) -> List[Point]:
    """Generate galaxy-like initial conditions with an exponential disk
    and a small Gaussian bulge.

    The disk surface density follows Sigma ~ exp(-r / R_d).
    Radii are sampled via the Erlang-2 trick: ``r = R_d * (E1 + E2)``
    where E1, E2 are independent exponentials.

    Parameters
    ----------
    n : int
        Total number of particles.
    R_d : float
        Disk scale length.
    R_max : float
        Maximum radius (particles beyond this are clipped).
    M_total : float
        Total mass of the system.
    G : float
        Gravitational constant used for circular-velocity estimates.
    bulge_frac : float
        Fraction of particles placed in the central bulge.
    bulge_sigma : float
        Standard deviation of the bulge Gaussian.
    rotation : {+1, -1}
        Sign of rotation (+1 = counter-clockwise).
    v_dispersion : float
        Velocity dispersion added to circular velocities.
    center : (float, float)
        Center of the disk in the 2-D plane.
    seed : int
        Random seed.

    Returns
    -------
    list[Point]
    """
    rng = np.random.default_rng(seed)
    n_bulge = int(bulge_frac * n)
    n_disk = n - n_bulge
    m = M_total / n
    cx, cy = center
    pts: List[Point] = []

    # Bulge
    if n_bulge > 0:
        xb = rng.normal(cx, bulge_sigma, n_bulge)
        yb = rng.normal(cy, bulge_sigma, n_bulge)
        vxb = rng.normal(0.0, v_dispersion, n_bulge)
        vyb = rng.normal(0.0, v_dispersion, n_bulge)
        for i in range(n_bulge):
            pts.append(Point(float(xb[i]), float(yb[i]), mass=m,
                             vx=float(vxb[i]), vy=float(vyb[i])))

    # Disk
    e1 = rng.exponential(1.0, n_disk)
    e2 = rng.exponential(1.0, n_disk)
    r = R_d * (e1 + e2)
    r = np.clip(r, 0.0, R_max)
    th = rng.uniform(0, 2 * np.pi, n_disk)
    xd = cx + r * np.cos(th)
    yd = cy + r * np.sin(th)

    # Approximate enclosed mass (disk + bulge)
    M_disk = M_total * (1 - bulge_frac)
    M_bulge = M_total * bulge_frac
    a = 2.5 * bulge_sigma + 1e-6
    M_enc = (
        M_disk * (1 - np.exp(-r / R_d) * (1 + r / R_d))
        + M_bulge * (r ** 2 / (r ** 2 + a ** 2))
    )
    v_c = np.sqrt(np.maximum(0.0, G * M_enc / np.maximum(r, 0.5)))

    tx = rotation * (-np.sin(th))
    ty = rotation * np.cos(th)
    vxd = v_c * tx + rng.normal(0.0, v_dispersion, n_disk)
    vyd = v_c * ty + rng.normal(0.0, v_dispersion, n_disk)

    for i in range(n_disk):
        pts.append(Point(float(xd[i]), float(yd[i]), mass=m,
                         vx=float(vxd[i]), vy=float(vyd[i])))
    return pts


def make_plummer_sphere(
    n: int = 500,
    M_total: float = 100.0,
    a: float = 10.0,
    G: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    seed: int = 42,
) -> List[Point]:
    """Generate particles sampled from a Plummer density profile
    (projected into 2-D).

    The Plummer model has density rho(r) ~ (1 + r^2/a^2)^{-5/2} and
    potential Phi(r) = -G M / sqrt(r^2 + a^2).

    Velocities are set to approximate virial equilibrium.

    Parameters
    ----------
    n : int
        Number of particles.
    M_total : float
        Total mass.
    a : float
        Plummer scale radius.
    G : float
        Gravitational constant.
    center : (float, float)
        Center position.
    seed : int
        Random seed.

    Returns
    -------
    list[Point]
    """
    rng = np.random.default_rng(seed)
    m = M_total / n
    cx, cy = center

    # Sample radii from Plummer CDF: M(<r)/M = r^3 / (r^2 + a^2)^{3/2}
    # Inverse CDF: r = a / sqrt(u^{-2/3} - 1)
    u = rng.uniform(0.01, 1.0, n)
    r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)

    # Random angles
    th = rng.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(th)
    y = cy + r * np.sin(th)

    # Velocity from virial: v_circ ~ sqrt(G M(<r) / r)
    M_enc = M_total * r ** 3 / (r ** 2 + a ** 2) ** 1.5
    v_c = np.sqrt(np.maximum(0.0, G * M_enc / np.maximum(r, 0.1 * a)))

    # Give each particle a tangential velocity with some dispersion
    tx = -np.sin(th)
    ty = np.cos(th)
    dispersion = 0.1 * v_c
    vx = v_c * tx + rng.normal(0.0, 1.0, n) * dispersion
    vy = v_c * ty + rng.normal(0.0, 1.0, n) * dispersion

    return [
        Point(float(x[i]), float(y[i]), mass=m,
              vx=float(vx[i]), vy=float(vy[i]))
        for i in range(n)
    ]


def make_random_uniform(
    n: int = 100,
    width: float = 100.0,
    height: Optional[float] = None,
    mass_range: Tuple[float, float] = (1.0, 1.0),
    v_max: float = 0.1,
    center: Tuple[float, float] = (0.0, 0.0),
    seed: int = 42,
) -> List[Point]:
    """Generate *n* particles uniformly distributed in a rectangle.

    Parameters
    ----------
    n : int
        Number of particles.
    width, height : float
        Box dimensions (height defaults to width if None).
    mass_range : (float, float)
        Uniform mass range ``[low, high]``.
    v_max : float
        Maximum initial velocity component.
    center : (float, float)
        Box center.
    seed : int
        Random seed.

    Returns
    -------
    list[Point]
    """
    if height is None:
        height = width
    rng = np.random.default_rng(seed)
    cx, cy = center
    hw, hh = width / 2, height / 2
    x = rng.uniform(cx - hw, cx + hw, n)
    y = rng.uniform(cy - hh, cy + hh, n)
    masses = rng.uniform(mass_range[0], mass_range[1], n)
    vx = rng.uniform(-v_max, v_max, n)
    vy = rng.uniform(-v_max, v_max, n)
    return [
        Point(float(x[i]), float(y[i]), mass=float(masses[i]),
              vx=float(vx[i]), vy=float(vy[i]))
        for i in range(n)
    ]


def make_solar_system(scale: float = 1.0) -> List[Point]:
    """Return a simplified 2-D solar system (Sun + 8 planets).

    Positions and velocities are approximate and use SI-like units
    scaled by *scale*.  Useful as a pedagogical example.

    Returns
    -------
    list[Point]
    """
    G_SI = 6.674e-11
    AU = 1.496e11  # meters

    # (name, mass_kg, dist_AU, v_m_s)
    bodies = [
        ("Sun",     1.989e30,  0.0,      0.0),
        ("Mercury", 3.285e23,  0.387,    47360),
        ("Venus",   4.867e24,  0.723,    35020),
        ("Earth",   5.972e24,  1.000,    29780),
        ("Mars",    6.390e23,  1.524,    24070),
        ("Jupiter", 1.898e27,  5.203,    13070),
        ("Saturn",  5.683e26,  9.537,    9690),
        ("Uranus",  8.681e25,  19.19,    6810),
        ("Neptune", 1.024e26,  30.07,    5430),
    ]

    pts: List[Point] = []
    for name, mass, dist_au, v in bodies:
        x = dist_au * AU * scale
        vy = v * scale
        pts.append(Point(x=x, y=0.0, mass=mass, vx=0.0, vy=vy))
    # Sun gets no net velocity initially
    return pts


# =====================================================================
#  Utility
# =====================================================================


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


# =====================================================================
#  CLI entry-point (demo)
# =====================================================================


if __name__ == "__main__":
    print(f"barnes_hut v{__version__}")
    print("Running demo: 200-particle exponential disk...")

    pts = make_exponential_disk(n=200, G=0.1, seed=42)
    sim = Simulation(pts, G=0.1, theta=0.6, eps=0.15, dt=0.06)

    print(f"  Initial KE = {sim.kinetic_energy():.4f}")
    print(f"  Initial momentum = {sim.total_momentum()}")

    sim.run(
        n_steps=100,
        show=True,
        viewport=(-45, 45, -45, 45),
        point_size=3,
    )

    print(f"  Final KE = {sim.kinetic_energy():.4f}")
    print(f"  Final momentum = {sim.total_momentum()}")
    print("Done.")
