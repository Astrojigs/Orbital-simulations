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

        ne = Rectangle(x + w/4,y - h/4, w/2,h/2)
        self.northeast = Quadtree(ne,self.capacity);
        nw = Rectangle(x - w/4,y - h/4, w/2,h/2)
        self.northwest = Quadtree(nw,self.capacity);
        se = Rectangle(x + w/4,y + h/4, w/2,h/2)
        self.southeast = Quadtree(se,self.capacity);
        sw = Rectangle(x - w/4,y + h/4, w/2,h/2)
        self.southwest = Quadtree(sw,self.capacity);

        self.divided=True

    def insert(self,point):
        if self.boundary.contains(point) != True:
            print("point not in boundary")
            return

        print("point is in boundary!")
        if (len(self.points) < self.capacity):
            print('point is add')
            self.points.append(point)

        # If there is no division yet, then divide
        if self.divided != True:
            self.subdivide()
            self.divided=True

            self.northeast.insert(point)
            self.northwest.insert(point)
            self.southeast.insert(point)
            self.southwest.insert(point)

    def show(self,axis):
        axis.add_patch(patches.Rectangle((self.boundary.x, self.boundary.y),
        self.boundary.w*2, self.boundary.h*2,
         fill=False))

        if self.divided:
            self.northeast.show(axis)
            self.northwest.show(axis)
            self.southeast.show(axis)
            self.southwest.show(axis)
        count=0
        for p in self.points:
            count+=1
            print(count)
            axis.scatter(p.x,p.y,c='black')
