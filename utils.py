import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        return (point.x >= self.x-self.w and point.x <= self.x + self.w and
        point.y >= self.y - self.h and point.y <= self.y + self.h)

    def show(self, axis):
        axis.add_patch(patches.Rectangle((self.x-self.w,self.y-self.h),self.w*2,self.h*2,fill=False))

class Quadtree:
    def __init__(self,boundary,n):

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
        w = self.boundary.w
        h = self.boundary.h

        ne = Rectangle(x + w/2,y - h/2, w,h)
        self.northeast = Quadtree(ne,self.capacity);
        nw = Rectangle(x - w/2,y - h/2, w,h)
        self.northwest = Quadtree(nw,self.capacity);
        se = Rectangle(x + w/2,y + h/2, w,h)
        self.southeast = Quadtree(se,self.capacity);
        sw = Rectangle(x - w/2,y + h/2, w,h)
        self.southwest = Quadtree(sw,self.capacity);

        self.divided=True

    def insert(self,point):
        if self.boundary.contains(point) != True:
            return

        if (len(self.points) < self.capacity):
            self.points.append(point)
        # If there is no division yet, then divide
        if not self.divided:
            self.subdivide()

        self.northeast.insert(point)
        self.northwest.insert(point)
        self.southeast.insert(point)
        self.southwest.insert(point)
    def show(self,axis):
        axis.add_patch(patches.Rectangle((self.boundary.x-self.boundary.w, self.boundary.y-self.boundary.h),
        self.boundary.w*2, self.boundary.h*2,
         fill=False))

        self.boundary.show(axis)
        if self.divided:

            self.northeast.show(axis)
            self.northwest.show(axis)
            self.southeast.show(axis)
            self.southwest.show(axis)
        count=0
        for p in self.points:
            count+=1
            print(count)
