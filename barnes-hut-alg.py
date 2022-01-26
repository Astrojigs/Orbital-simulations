import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import utils

boundary = utils.Rectangle(200,200,200,200)
qt = utils.Quadtree(boundary, 4)

for i in range(50):
    p = utils.Point(np.random.uniform(0,400),np.random.uniform(0,400))
    qt.insert(p)
