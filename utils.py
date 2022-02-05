import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Point:
    def __init__(self,x,y, mass=1,vx=0,vy=0):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = vx
        self.vy = vy
class Rectangle:
    def __init__(self,x,y,w,h):
        # x,y = center of the Rectangle
        # w = edge to edge horizontal distance
        # h = edge to edge vertical distance
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.west_edge, self.east_edge = x - w/2, x + w/2
        self.north_edge, self.south_edge = y-h/2, y+h/2

    def contains(self,point):
        return (point.x >= self.west_edge and point.x <= self.east_edge and
        point.y >= self.north_edge and point.y <= self.south_edge)

    def intersects(self,other):
        """Does the other Rectangle object intersect with this one?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def show(self, axis,color='black'):
        #axis.add_patch(patches.Rectangle((self.x-self.w,self.y-self.h),self.w*2,self.h*2,fill=False))
        x1, y1 = self.west_edge,self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        # axis.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c='black', lw=1)
        axis.add_patch(patches.Rectangle((self.west_edge,self.south_edge),
        (self.east_edge-self.west_edge),
        (self.north_edge-self.south_edge),fill=False,color=color))
class Quadtree:
    Mass = 0
    CenterOfMass_x = 0
    CenterOfMass_y = 0
    def __init__(self,boundary,n = 4):

        self.boundary = boundary

        # choosing capacity(n) = 4, i.e. if particle number crosses 4 than sub-divide
        # When do i choose that i need to sub-divide
        self.capacity = n

        # Keep track of points:
        self.points = []

        self.divided = False

    def subdivide(self):

        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w/2
        h = self.boundary.h/2

        ne = Rectangle(x + w/2,y - h/2, w,h)
        self.northeast = Quadtree(ne,self.capacity);
        nw = Rectangle(x - w/2,y - h/2, w,h)
        self.northwest = Quadtree(nw,self.capacity);
        se = Rectangle(x + w/2,y + h/2, w,h)
        self.southeast = Quadtree(se,self.capacity);
        sw = Rectangle(x - w/2,y + h/2, w,h)
        self.southwest = Quadtree(sw,self.capacity);

        self.divided = True

    def insert(self,point):

        # If the point isn't in the boundary then stop!
        if self.boundary.contains(point) != True:
            return False

        # Check if the number of points exceed the capacity
        if len(self.points) < self.capacity:
            # if the point does not exceed then add the point,
            # to the list of points in the boundary
            self.points.append(point)
            return True
        # If the number of points exceed the given capacity then
        # subdivide the rectangular boundary into four parts

        # subdivide boundary
        if not self.divided:
            self.subdivide()

        return (self.northeast.insert(point) or
        self.northwest.insert(point) or
        self.southeast.insert(point) or
        self.southwest.insert(point))


    def query(self, boundary, found_points):
        """Find points in the quadtree that lie within a boundary."""
        if not self.boundary.intersects(boundary):
            # if the domain of this node does not interesect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's point to see if they lie within boundary
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # if this node has children, search them too.
        if self.divided:
            self.northeast.query(boundary,found_points)
            self.northwest.query(boundary,found_points)
            self.southeast.query(boundary,found_points)
            self.southwest.query(boundary,found_points)
        return found_points

    def compute_mass_distribution(self):
        Mass = 0
        CenterOfMass_x = 0
        CenterOfMass_y = 0

        if len(self.points) <= 4 and len(self.points)!=0:
            for i in range(len(self.points)):
                CenterOfMass_x += self.points[i].x*self.points[i].mass
                CenterOfMass_y += self.points[i].y*self.points[i].mass
                Mass += self.points[i].mass
                return Mass, CenterOfMass_x/Mass, CenterOfMass_y/Mass

        else:
            # Compute the center of mass based on the masses
            # of all child quadrants and the center of mass as
            # the center of mass of the child quadrants weights with their mass
            if self.divided:
                ne_mass,ne_com_x,ne_com_y = self.northeast.compute_mass_distribution()
                nw_mass,nw_com_x,nw_com_y = self.northwest.compute_mass_distribution()
                se_mass,se_com_x,se_com_y = self.southeast.compute_mass_distribution()
                sw_mass,sw_com_x,sw_com_y = self.southwest.compute_mass_distribution()
                Mass = ne_mass + nw_mass + se_mass + sw_mass
                CenterOfMass_x = ne_mass*ne_com_x + nw_mass*nw_com_x + se_mass*se_com_x + sw_mass*se_com_x
                CenterOfMass_y = ne_mass*ne_com_y + nw_mass*nw_com_y + se_mass*se_com_y + sw_mass*se_com_y

                return Mass, CenterOfMass_x/Mass, CenterOfMass_y/Mass

    def distance(x1,y1,x2,y2):
        return (np.sqrt((x2-x1)**2 + (y2-y1)**2))

    def calculate_force(self, point,G=0.1,theta=1.1):
        force = 0
        if len(self.points) <=4 and len(self.points)!=0:
            for i in range(len(self.points)):
                force += G*self.points[i].mass*point.mass/self.distance(self.points[i].x,self.points[i].y,
                point.x,point.y)
        else:
            mass_node, r_x,r_y = self.compute_mass_distribution()
            r = np.sqrt(r_x**2 + r_y**2)
            d = abs(self.north_edge - self.south_edge)
            if d/r <theta:
                force = G*point.mass*self.compute_mass_distribution()[0]/self.distance(point.x,point.y,
                self.compute_mass_distribution()[1],self.compute_mass_distribution()[2])
            else:
                # Compute force on child node
                if self.divided:
                    self.northwest.calculate_force(self, point, G, theta)
                    self.northeast.calculate_force(self, point, G, theta)
                    self.southwest.calculate_force(self, point, G, theta)
                    self.southeast.calculate_force(self, point, G, theta)

    def show(self,axis):
        self.boundary.show(axis)
        if self.divided:
            self.northeast.show(axis)
            self.northwest.show(axis)
            self.southeast.show(axis)
            self.southwest.show(axis)
