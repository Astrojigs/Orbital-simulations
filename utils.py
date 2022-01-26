import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self,point):
        if (point.x > self.x-self.w and point.x < self.x + self.w and
        point.y > self.y - self.h and point.y < self.y + self.h):
            return True
        else:
            return False


class Quadtree:
    def __init__(self,boundary,n):

        self.boundary = boundary

        self.x = self.boundary.x
        self.y = self.boundary.y
        self.w = self.boundary.w
        self.h = self.boundary.h

        # choosing capacity(n) = 4, i.e. if particle number crosses 4 than sub-divide
        # When do i choose that i need to sub-divide
        self.capacity = n

        # Keep track of points:
        self.points = []

        self.divided = False

    def subdivide(self):

        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w
        h = self.boundary.h

        ne = Rectangle(x + w/2,y - h/2,w/2,h/2)
        nw = Rectangle(x - w/2,y - h/2,w/2,h/2)
        se = Rectangle(x + w/2,y + h/2,w/2,h/2)
        sw = Rectangle(x - w/2,y + h/2,w/2,h/2)

        self.northwest = Quadtree(nw,self.capacity);
        self.northeast = Quadtree(ne,self.capacity);
        self.southeast = Quadtree(se,self.capacity);
        self.southwest = Quadtree(sw,self.capacity);

    def insert(self,point):
        if self.boundary.contains(point) == False:
            return


        if (len(self.points) < self.capacity):
            self.points.append(point)
        else:
            if (self.divided == False):
                self.subdivide()
                self.divided=True
                self.northeast.insert(point)
                self.northwest.insert(point)
                self.southeast.insert(point)
                self.southwest.insert(point)

    def show(self):
        Rectangle(self.boundary.x, self.boundary.y,self.boundary.w/2, self.boundary.h/2)
        if (self.divided):
            self.northeast.show()
            self.northwest.show()
            self.southeast.show()
            self.southwest.show()
