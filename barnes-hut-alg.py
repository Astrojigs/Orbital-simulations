import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import utils
fig, ax = plt.subplots(figsize=(5,5))

# specs of Rectangle:
width = 400
height = 400
center_x = 200
center_y = 200
boundary = utils.Rectangle(center_x,center_y,width,height)
qt = utils.Quadtree(boundary, 4)

for i in range(100):
    x = np.random.randint(0,400)
    y = np.random.randint(0,400)
    p = utils.Point(x,y)
    ax.scatter(p.x,p.y,s=1,c='black')
    qt.insert(p)

print(f"mass = {qt.compute_mass_distribution()}")
ax.scatter(qt.compute_mass_distribution()[1],qt.compute_mass_distribution()[2],s=100,c='blue')
# give me points in this region
region = utils.Rectangle(100,100,108,50)
found_points = []
qt.query(region,found_points)
ax.scatter([p.x for p in found_points],[p.y for p in found_points],
facecolors='none', edgecolors='r',s=32)
region.show(ax,color='red')
ax.set_xlim(center_x-width/2,center_x+width/2)
ax.set_ylim(center_y-height/2,center_y+height/2)

qt.show(ax)
plt.show()
