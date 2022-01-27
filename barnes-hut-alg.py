import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import utils
fig, ax = plt.subplots(figsize=(5,5))

boundary = utils.Rectangle(200,200,200,200)
qt = utils.Quadtree(boundary, 4)

for i in range(10):
    p = utils.Point(np.random.uniform(0,400),np.random.uniform(0,400))
    ax.scatter(p.x,p.y,s=1,c='black')
    qt.insert(p)

ax.set_xlim(0,400)
ax.set_ylim(0,400)

qt.show(ax)
plt.show()
