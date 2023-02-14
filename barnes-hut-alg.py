import utils
import matplotlib.pyplot as plt
import numpy as np
height= 100
width = 100
center_x = 50
center_y = 50

# Creating a boundary instance:
axis = None
boundary = utils.Rectangle(center_x,center_y,width,height)

# Create points:
points = [utils.Point(x=np.random.normal(loc=50,scale=20),
y=np.random.normal(loc=50,scale=20),mass=1) for i in range(100)]

# Create a quadtree instance:
qt = utils.Quadtree(boundary,G=1,theta_=1)

for p in points:
    qt.insert(p)

# plt.scatter([p.x for p in points],[p.y for p in points],s=5)
# qt.show()
# plt.show()
print(len(qt.points))
