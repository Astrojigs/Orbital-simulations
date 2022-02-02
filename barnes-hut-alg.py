import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import utils
fig, ax = plt.subplots(figsize=(5,5))

boundary = utils.Rectangle(200,200,400,400)
qt = utils.Quadtree(boundary, 4)

for i in range(100):
    x = np.random.randint(0,400)
    y = np.random.randint(0,400)

    p = utils.Point(x,y)
    ax.scatter(p.x,p.y,s=1,c='black')
    qt.insert(p)
ax.set_xlim(-100,500)
ax.set_ylim(-100,500)

qt.show(ax)
plt.show()
