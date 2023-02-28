"""
Property Of Jigar Patel
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import astropy
from astropy import constants as astropy_constants

class constants:
    """
    Stores constants

    constants.info() : gives information about each term in units being used:

    Use the following terms in the corresponding 1 unit:

    distance : 1 Light year
    time : 1 Million year
    Mass : Solar Masses
    G : 0.156079 ly^3/(solar_mass x million_year^2)
    """

    G = astropy_constants.G.to_value((astropy.units.lyr.decompose())**3/((astropy.units.M_sun.decompose())*(np.square((1000000*astropy.units.yr).decompose()))))
    def info():
        print("distance : 1 Light year\ntime : 1 Million year\nMass : Solar Masses\nG : 0.156079 ly^3/(solar_mass x million_year^2)")

class Point:
    """
    Create a point in space.
    """
    def __init__(self,x,y, mass=1.0,vx=0,vy=0, acc_x=0, acc_y=0, color=None):
        """
        Each point will have:
        mass (default 1.0),
        vx = velocity x_component (default = 0),
        vy = velocity y_component (default = 0),
        acc_x = acceleration x_component (default = 0),
        acc_y = acceleration y_component (default = 0)
        """

        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.color = color

    def update_position(self, quadtree,dt):
        """
        Update x,y component of position and velocity of a point.

        Details:
        Uses returned value from
        `quadtree.calculate_force()` method
        """

        # Calculate the net force on the point
        force_x, force_y = quadtree.calculate_force(self)
        # Update the position
        self.vx += force_x / self.mass
        self.vy += force_y / self.mass
        self.x += self.vx*dt
        self.y += self.vy*dt

class Rectangle:
    """
    Create a boundary (rectangle) within which a Quadtree can function.
    """
    def __init__(self,x,y,w,h):
        """
        x = center of the Rectangle
        y = center of the Rectangle
        w = width of the rectangle
        h = height of the rectangle
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.west_edge, self.east_edge = x - w/2, x + w/2
        self.north_edge, self.south_edge = y + h/2, y - h/2

    def contains(self,point):
        """
        Check if point exists within boundary.
        """
        return (point.x >= self.west_edge and point.x <= self.east_edge and
        point.y <= self.north_edge and point.y >= self.south_edge)

    def show(self, axis,color='red'):
        """
        Shows the rectangle boundary.
        Default color : red

        Details:
        Uses axis.plot() method to plot the boundary.
        If axis = None then uses plt.gca()
        """
        if axis == None:
            axis = plt.gca()
        x1, y1 = self.west_edge,self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        axis.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=color, lw=1)



class Quadtree:
    """
    Create a quadtree in the given Boundary (`Rectangle` object).
    """
    def __init__(self,boundary, G,theta_, n = 1):
        """
        Parameters:

        boundary: Rectangle instance
        n = capacity
            choosing n = 1, i.e. if particle number crosses 1 than sub-divide
        G = gravitational constant
        theta_ = barnes hut algorithm theta (default value = 1)
        """
        self.boundary = boundary

        # choosing capacity(n) = 1, i.e. if particle number crosses 1 than sub-divide
        self.capacity = n

        # Keep track of points:
        self.points = []
        self.quads = [None, None, None, None]
        self.divided = False
        self.mass = 0.0
        self.G = G
        self.theta_ = theta_
        self.center_of_mass_x = 0.0
        self.center_of_mass_y = 0.0



    def create_quadtree(self):
        """
        Returns a new quadtree"""
        return Quadtree(self.boundary,self.G,self.theta_,n=self.capacity)

    def subdivide(self):
        """
        Subdivides the region into four parts
               1  |  2
             _____|____
                  |
               3  |  4

               quad[0] = 1st quadrant (north west)
               quad[1] = 2nd quadrant (north east)
               quad[2] = 3rd quadrant (south west)
               quad[3] = 4th quadrant (south east)

        """
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w/2
        h = self.boundary.h/2

        ne = Rectangle(x + w/2, y + h/2, w, h)
        self.quads[1] = Quadtree(ne, G= self.G, theta_=self.theta_, n=self.capacity)
        nw = Rectangle(x - w/2, y + h/2, w, h)
        self.quads[0] = Quadtree(nw, G= self.G, theta_=self.theta_, n=self.capacity)
        se = Rectangle(x + w/2, y - h/2, w, h)
        self.quads[3] = Quadtree(se, G= self.G, theta_=self.theta_, n=self.capacity)
        sw = Rectangle(x - w/2, y - h/2, w, h)
        self.quads[2] = Quadtree(sw, G= self.G, theta_=self.theta_, n=self.capacity)

        self.divided = True

        # Check if existing points contain in children:
        for point in self.points:
            for quad in self.quads:
                quad.insert(point)


    def limit_reached(self):
        """
        Checks the recursive limit and returns Boolean value

        Returns: False (if limit not reached)
        Returns: True (if limit reached (stop adding points))
        """
        threshold = 0.1

        if self.boundary.w < threshold:
            return True

        return False

    def insert(self, point):
        """
        Insert a point in the quadtree.

        Details:
        Each quadrant in the quadtree is assigned a list that contains
         all point-objects that lie within its boundary.
         While calling qt.insert(), the following logic flow is operated:

         if quadrant is divided:
             check each of the 4 children/node.
             if the point is inserted in the child:
                 1) Update the list of the parent quadrant by adding the inserted point.
                 2) Update the center of mass of the parent quadrant
             if point is in boundary:
                 if nothing in the quadrant:
                     update the quadrant.mass = point.mass
                     update the COM_x = point.mass*point.x
                     update the COM_y = point.mass*point.y
                     return True
                if Point already exists then:
                    COM_x += point.mass*point.x
                    COM_y += point.mass*point.y



                 """
        if self.divided:
            for quad in self.quads:
                if quad.insert(point):
                    self.points.append(point)
                    self.mass = sum([p.mass for p in self.points])
                    self.center_of_mass_x = sum([p.mass*p.x for p in self.points])
                    self.center_of_mass_y = sum([p.mass*p.y for p in self.points])
                    return True

        # Check if the point is in Boundary
        if self.boundary.contains(point):

            if len(self.points) < self.capacity and not self.limit_reached():
                self.points.append(point)
                self.mass = point.mass
                self.center_of_mass_x = point.x*point.mass
                self.center_of_mass_y = point.y*point.mass
                return True

            self.points.append(point)
            self.mass = sum([p.mass for p in self.points])
            self.center_of_mass_x = sum([p.mass*p.x for p in self.points])
            self.center_of_mass_y = sum([p.mass*p.y for p in self.points])

            if not self.divided and not self.limit_reached():
                self.subdivide()
                for quad in self.quads:
                    if len(quad.points) == 0:
                        quad.insert(point)
                        return True
        else:
            return False


    def center_of_mass(self):
        """
        Returns (X, Y) for COM
        """
        return self.center_of_mass_x/self.mass, self.center_of_mass_y/self.mass

    def calculate_force(self, point):
        if self.mass == 0 :
            #print("self.mass = 0")
            return 0,0
        if not self.divided:
            #print("Not divided")
            #print("Using point force calculations:")
            return self._calculate_force_on_point(point)
        else:
            #print(f"It is divided.")
            force_x, force_y = 0,0
            for quad in self.quads:
                #print(f"\nInspecting quad with {quad.boundary.x,quad.boundary.y} center")
                if len(quad.points) != 0:
                    if quad._should_use_approximation(point):
                        #print("Using approximation")
                        fx,fy = quad._calculate_force_on_point_approximation(point)
                        force_x += fx
                        force_y += fy
                    else:
                        #print("Not using approximation")
                        fx, fy = quad.calculate_force(point)
                        force_x += fx
                        force_y += fy
            #print(f"Force = {force_x, force_y}")
            return force_x,force_y

    def _calculate_force_on_point(self, point):
        """Calculates the force on the point due to all other points in the quadtree"""
        #print("Used point force calculation")
        force_x, force_y = 0, 0
        for other_point in self.points:
            if other_point != point:
                dx = other_point.x - point.x
                dy = other_point.y - point.y
                r = (dx ** 2 + dy ** 2) ** 0.5
                if r == 0:
                    continue
                force = self.G * point.mass * other_point.mass / (r ** 2)
                force_x += force * dx / r
                force_y += force * dy / r
        #print(f"force from using no approximation = {force_x,force_y}")
        return force_x, force_y

    def _calculate_force_on_point_approximation(self, point):
        """Calculates the force on the point due to the center of mass of the quadtree"""
        com_x,com_y = self.center_of_mass()
        #print(f"used approximation function: \n center of mass = {com_x,com_y}")
        dx = com_x - point.x
        dy = com_y - point.y
        r = (dx ** 2 + dy ** 2) ** 0.5
        if r == 0:
            return 0, 0
        force = self.G * point.mass * self.mass / (r ** 2)
        #print(f"Force from approximation: {force*dx/r,force*dy/r}")
        return force * dx / r, force * dy / r

    def _should_use_approximation(self, point):
        """
        Barnes Hut Algorithm condition.
        Returns: w/r < theta_.
        Details:
        w = width of the quadrant
        r is the distance from the point to the COM of quadrant.

        """
        com_x, com_y = self.center_of_mass()
        r = ((point.x - com_x)**2 + (point.y - com_y)**2)**0.5
        if r ==0 or len(self.points)==0:
            return False
        #print(f"self.boundary.w = {self.boundary.w} and self.theta = {self.theta_} \n ratio = {self.boundary.w/r}")
        return self.boundary.w / r < self.theta_


    def clear(self):
        """
        Clears the quadtree by resetting all points and sub-quadrants
        """
        self.points = []
        self.mass = 0.0
        self.center_of_mass_x = 0
        self.center_of_mass_y = 0
        self.divided = False
        self.quads = [None, None, None, None]

    def show(self, axis=None,show_entire=False):
        """
        Shows the quadtree

        Will use plt.gca() if no axis is given.

        show_entire = will also plot quadrants without any points (default = False)"""
        if axis == None:
            axis = plt.gca()
        if not show_entire:
            if len(self.points)!=0:
                self.boundary.show(axis)
                if self.divided:
                    for quad in self.quads:
                        quad.show(axis)
        elif show_entire:
            self.boundary.show(axis)
            if self.divided:
                for quad in self.quads:
                    quad.show(axis,show_entire=True)

    def show_from_point(self, point, axis=None, show_mass = False, color='red'):
        """Shows the quadtree w.r.t given point. Uses Barnes Hut algorithm.
        put temp = 1
        """
        if axis==None:
            axis=plt.gca()

        # create a qt for com
        com_qt = self.create_quadtree()
        if self.mass == 0:
            # do not insert
            pass
        if not self.divided:
            # insert point
            for p in self.points:
                if p != point:
                    x,y = self.center_of_mass()
                    com_mass = self.mass
                    com_point = Point(x,y,mass=com_mass)
                    com_qt.insert(com_point)
                    #com_qt.insert(p)
        else:
            for quad in self.quads:
                if len(quad.points) != 0:
                    if quad._should_use_approximation(point):
                        x,y = quad.center_of_mass()
                        com_mass = quad.mass
                        com_point = Point(x,y,mass=com_mass)
                        com_qt.insert(com_point)
                    else:
                        #print("Not using approximation")
                        quad.show_from_point(point,axis)
                        quad.insert(point)
        if show_mass:
            px,py = [],[]
            print(com_qt.points)
            for p in com_qt.points:
                print(p)
                px.append(p.x)
                py.append(p.y)
            axis.scatter(px,py,c='brown',s=100)
        com_qt.show(axis)

def get_cart_coords(r,theta):
    """
    Convert r, theta to x,y
    """
    return r*np.cos(np.deg2rad(theta)), r*np.sin(np.deg2rad(theta))

def get_cart_coords_vel(rv, thetav, r, theta):
    """
    Convert velocities:
    [dr/dt, dtheta/dt] to [dx/dt,dy/dt]
    """
    vx = rv*np.cos(np.deg2rad(theta)) - r*np.sin(np.deg2rad(theta))*thetav
    vy = rv*np.sin(np.deg2rad(theta)) + r*np.cos(np.deg2rad(theta))*thetav
    return vx, vy

def barnes_hut_sim(points, _wrap_points=False,
                   dt=1, g_const=0, theta=1,
                   save_to_video = False,
                   show_quadtree=False, show_quadtree_wrt_point = None,
                  n_frames=50, add_quadtree_plot=False):
    """
    """
    def wrap_point(p):
        """
        Wraps the points such that they don't leave the boundary.
        """
        # wrap around when the particle goes out of bounds
        if p.x > width:
            p.x -= width
        elif p.x < 0:
            p.x += width
        if p.y > height:
            p.y -= height
        elif p.y < 0:
            p.y += height

    qt = Quadtree(boundary,n=1, G=g_const, theta_=theta)
    for p in points:
        qt.insert(p)
    frames = []

    for time in range(n_frames):
        if not add_quadtree_plot:
            fig, ax = plt.subplots(figsize=(10,10))
        else:
            fig, ax = plt.subplots(1,2,figsize=(15,7))
        ps = []
        qt.clear()

        for p in points:
            if _wrap_points:
                wrap_point(p)

            p.update_position(qt,dt=dt)
            qt.insert(p)

            ps.append([p.x,p.y])

        ps = np.array(ps)


        if not add_quadtree_plot:
            ax.scatter(points[0].x,points[0].y,c='orange',s=40)
            ax.scatter(ps[:,0],ps[:,1],s=1,alpha=0.5,c='black')
            ax.set_xlim(-width*0.1, width*1.1)
            ax.set_ylim(-height*0.1, height*1.1)

        else:
            ax[0].scatter(ps[:,0],ps[:,1],s=10,alpha=0.5,c='black')
            ax[0].set_xlim(-width*0.1, width*1.1)
            ax[0].set_ylim(-height*0.1, height*1.1)
            show_quadtree = False
            ax[1].scatter(ps[:,0],ps[:,1],s=10,alpha=0.5,c='black')
            ax[1].set_xlim(-width*0.1, width*1.1)
            ax[1].set_ylim(-height*0.1, height*1.1)
            qt.show(ax[1])

        if show_quadtree:
            # Show the quadtree
            qt.show(ax)
        if show_quadtree_wrt_point != None:
            qt.show_from_point(show_quadtree_wrt_point, ax)
            ax.scatter(show_quadtree_wrt_point.x,show_quadtree_wrt_point.y,s=50,c='orange')
        # Convert the figure to a numpy array
        if save_to_video:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(data)


        #plt.show()
        print(f"frame: {time}/{n_frames}")
        clear_output(wait=True)
    if save_to_video:
        import imageio
        imageio.mimsave(save_to_video, frames, 'MP4', fps=30)

def plummer_density_profile_with_mass(n, r_scale, mass,G, center=(0, 0)):
    """
    Returns x and y coordinates, and x and y velocities of n stars using the Plummer density profile with mass

    Parameters:
    n (int): Number of stars to generate
    r_scale (float): Scale radius of the Plummer density profile
    mass (float): Mass of the central object
    center (tuple): (x, y) position of the center of the Plummer sphere

    Returns:
    x, y, vx, vy (np.ndarray): x and y coordinates, and x and y velocities of the stars
    """
    r = np.random.uniform(0, r_scale, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = center[0] + r * np.cos(theta) / (1 + (r/r_scale)**2)**0.5
    y = center[1] + r * np.sin(theta) / (1 + (r/r_scale)**2)**0.5
    v_circ = (G * mass / r)**0.5
    vx = v_circ * np.sin(theta)
    vy = -1 * v_circ * np.cos(theta)
    return x, y, vx, vy
