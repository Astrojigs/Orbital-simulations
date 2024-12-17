import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
from astropy import constants as astropy_constants
import astropy.units as u
import cv2


# class constants:
#     """
#     A class to store and display constants for the simulation.
#
#     Attributes:
#         G (float): Gravitational constant in ly^3 / (solar_mass x million_year^2)
#     """
#     G = astropy_constants.G.to_value((u.lyr.decompose()) ** 3 / (u.M_sun * (1e6 * u.yr) ** 2))
#
#     def info():
#         """
#         Displays the units used in the simulation.
#         """
#         print(
#             "distance: 1 Light year\ntime: 1 Million year\nMass: Solar Masses\nG: 0.156079 ly^3/(solar_mass x million_year^2)")


class Point:
    """
    Represents a point (particle/star) in 2D space with position, velocity, and mass.

    Attributes:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        mass (float): Mass of the point.
        vx (float): Velocity in the x-direction.
        vy (float): Velocity in the y-direction.
        acc_x (float): Acceleration in the x-direction.
        acc_y (float): Acceleration in the y-direction.
        color (str, optional): Color of the point for visualization.
    """

    def __init__(self, x, y, mass=1.0, vx=0, vy=0, acc_x=0, acc_y=0, color=None):
        self.x, self.y = x, y
        self.mass = mass
        self.vx, self.vy = vx, vy
        self.acc_x, self.acc_y = acc_x, acc_y
        self.color = color

    def update_position(self, quadtree, dt):
        """
        Updates the position and velocity of the point using gravitational forces.

        Args:
            quadtree (Quadtree): The quadtree used to calculate gravitational forces.
            dt (float): Time step for the simulation.
        """
        force_x, force_y = quadtree.calculate_force(self)
        self.vx += (force_x / self.mass) * dt
        self.vy += (force_y / self.mass) * dt
        self.x += self.vx * dt
        self.y += self.vy * dt


class Rectangle:
    """
    Represents a rectangular boundary in 2D space.

    Attributes:
        x (float): X-coordinate of the rectangle's center.
        y (float): Y-coordinate of the rectangle's center.
        w (float): Width of the rectangle.
        h (float): Height of the rectangle.
        west (float): Western edge.
        east (float): Eastern edge.
        north (float): Northern edge.
        south (float): Southern edge.
    """

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.west, self.east = x - w / 2, x + w / 2
        self.north, self.south = y + h / 2, y - h / 2

    def contains(self, point):
        """
        Checks if a point is within the rectangle.

        Args:
            point (Point): The point to check.

        Returns:
            bool: True if the point is within the rectangle, False otherwise.
        """
        return self.west <= point.x <= self.east and self.south <= point.y <= self.north

    def show(self, axis, color='red'):
        """
        Visualizes the rectangle boundary on a matplotlib axis.

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
            color (str): Color of the boundary (default: 'red').
        """
        axis.plot([self.west, self.east, self.east, self.west, self.west],
                  [self.north, self.north, self.south, self.south, self.north], c=color, lw=1)


class Quadtree:
    """
    Implements a quadtree data structure for the Barnes-Hut algorithm.

    Attributes:
        boundary (Rectangle): Boundary of the quadtree.
        G (float): Gravitational constant.
        theta_ (float): Barnes-Hut approximation threshold.
        capacity (int): Maximum points before subdividing.
        points (list): List of points within the quadtree.
        quads (list): Sub-quadrants of the quadtree.
        divided (bool): Whether the quadtree has been subdivided.
        mass (float): Total mass of points in the quadtree.
        center_of_mass_x (float): X-coordinate of the center of mass.
        center_of_mass_y (float): Y-coordinate of the center of mass.
    """

    def __init__(self, boundary, G, theta_, n=1):
        self.boundary, self.G, self.theta_ = boundary, G, theta_
        self.capacity = n
        self.points = []
        self.quads = [None] * 4
        self.divided = False
        self.mass, self.center_of_mass_x, self.center_of_mass_y = 0, 0, 0

    def subdivide(self):
        """
        Subdivides the quadtree into four quadrants.
        """
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w / 2, self.boundary.h / 2
        self.quads = [
            Quadtree(Rectangle(x - w / 2, y + h / 2, w, h), self.G, self.theta_),
            Quadtree(Rectangle(x + w / 2, y + h / 2, w, h), self.G, self.theta_),
            Quadtree(Rectangle(x - w / 2, y - h / 2, w, h), self.G, self.theta_),
            Quadtree(Rectangle(x + w / 2, y - h / 2, w, h), self.G, self.theta_)
        ]
        self.divided = True
        for p in self.points:
            for quad in self.quads:
                quad.insert(p)
        self.points.clear()

    def insert(self, point):
        """
        Inserts a point into the quadtree.

        Args:
            point (Point): The point to insert.

        Returns:
            bool: True if the point was inserted, False otherwise.
        """
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            self._update_mass_properties()
            return True

        if not self.divided:
            self.subdivide()
        for quad in self.quads:
            if quad.insert(point):
                return True
        return False

    def _update_mass_properties(self):
        """
        Updates the total mass and center of mass of the quadtree.
        """
        self.mass = sum(p.mass for p in self.points)
        if self.mass > 0:
            self.center_of_mass_x = sum(p.mass * p.x for p in self.points) / self.mass
            self.center_of_mass_y = sum(p.mass * p.y for p in self.points) / self.mass

    def calculate_force(self, point):
        """
        Calculates the gravitational force acting on a point.

        Args:
            point (Point): The point for which to calculate the force.

        Returns:
            tuple: Force in the x and y directions (fx, fy).
        """
        if self.mass == 0 or (len(self.points) == 1 and self.points[0] == point):
            return 0, 0
        dx, dy = self.center_of_mass_x - point.x, self.center_of_mass_y - point.y
        r = max((dx ** 2 + dy ** 2) ** 0.5, 1e-8)

        if not self.divided or self.boundary.w / r < self.theta_:
            force = self.G * point.mass * self.mass / r ** 2
            return force * dx / r, force * dy / r
        else:
            force_x, force_y = 0, 0
            for quad in self.quads:
                fx, fy = quad.calculate_force(point)
                force_x += fx
                force_y += fy
            return force_x, force_y

    def reset(self):
        """
        Clears all points in the quadtree and resets properties.
        """
        self.points.clear()
        self.mass, self.center_of_mass_x, self.center_of_mass_y = 0, 0, 0
        self.divided = False
        self.quads = [None] * 4

    def show(self, axis):
        """
        Visualizes the quadtree boundaries recursively on a matplotlib axis.

        Args:
            axis (matplotlib.axes.Axes): The axis to plot on.
        """
        self.boundary.show(axis)
        if self.divided:
            for quad in self.quads:
                quad.show(axis)


def barnes_hut_sim(points, width, height, dt=1, g_const=0.1, theta=0.85, n_frames=50, save_to_video=None):
    """
    Simulates gravitational interactions using the Barnes-Hut algorithm.

    Args:
        points (list): List of Point objects.
        width (float): Width of the simulation space.
        height (float): Height of the simulation space.
        dt (float): Time step for the simulation.
        g_const (float): Gravitational constant.
        theta (float): Barnes-Hut approximation threshold.
        n_frames (int): Number of frames for the simulation.
        save_to_video (str, optional): Path to save the simulation video.
    """
    boundary = Rectangle(width / 2, height / 2, width, height)
    qt = Quadtree(boundary, G=g_const, theta_=theta)
    frames = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_to_video, fourcc, 30, (1000, 1000)) if save_to_video else None

    for time in range(n_frames):
        qt.reset()
        for p in points:
            p.update_position(qt, dt)
            qt.insert(p)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter([p.x for p in points], [p.y for p in points], s=10, c='black')
        qt.show(ax)
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if save_to_video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # plt.close(fig)
        plt.show()
        print(f"Frame: {time}/{n_frames}")
        
        clear_output(wait=True)

    if save_to_video:
        out.release()


if __name__ == "__main__":
    n_stars = 100
    width, height = 100, 100
    points = [Point(np.random.rand() * width, np.random.rand() * height, mass=1.0) for _ in range(n_stars)]
    barnes_hut_sim(points, width, height, save_to_video="barnes_hut_sim.mp4")
