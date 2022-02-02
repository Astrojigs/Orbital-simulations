import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import utils
fig, ax = plt.subplots(figsize=(5,5))

# specs of Rectangle:
width = 400
height = 400
center_x = 0
center_y = 0
boundary = utils.Rectangle(center_x,center_y,width,height)
qt = utils.Quadtree(boundary, 4)

for i in range(5500):
    x = np.random.randint(0,400)
    y = np.random.randint(0,400)

    p = utils.Point(x,y)
    ax.scatter(p.x,p.y,s=1,c='black')
    qt.insert(p)
ax.set_xlim(center_x-width/2,center_x+width/2)
ax.set_ylim(center_y-height/2,center_y+height/2)

qt.show(ax)
plt.show()
