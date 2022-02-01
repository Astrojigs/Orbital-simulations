import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import utils
fig, ax = plt.subplots(figsize=(5,5))

boundary = utils.Rectangle(200,200,200,200)
qt = utils.Quadtree(boundary, 4)

for i in range(500):
    x = np.random.randint(0,400)
    y = np.random.randint(0,400)

    p = utils.Point(x,y)
    ax.scatter(p.x,p.y,s=1,c='black')
    qt.insert(p)
ax.set_xlim(-200,800)
ax.set_ylim(-200,800)

qt.show(ax)
plt.show()
