from astrojigs import *
import astrojigs
import matplotlib.pyplot as plt
import numpy as np
height= 100
width = 100
center_x = 50
center_y = 50

# Creating a boundary instance:
axis = None
boundary = astrojigs.Rectangle(center_x,center_y,width,height)


# Create points:
points = []
"""The Center of the galaxy has a mass of 1e4 solar mass
The rest of the stars will have masses ranging from 0.4, 1.8 solar masses."""
blackhole_point = Point(x=center_x,y=center_y, mass=1e4)
points.append(blackhole_point)
x,y,vx,vy = astrojigs.plummer_density_profile_with_mass(n=1000,r_scale=width*0.70,mass=blackhole_point.mass,G=astrojigs.constants.G,center=(center_x,center_y))
for i in range(len(x)):
    points.append(Point(x[i], y[i],
                        mass = np.random.uniform(0.4,1.8),
                        vx = vx[i], vy = vy[i]))

# Create a quadtree instance:
thetas = [0,0.5,0.75,1] # len = multiples of 2
fig, ax = plt.subplots(2,2,figsize=(10,10))

ax[0,0].set_title(f"Theta = {thetas[0]}, n^2 iterations")
qt = astrojigs.Quadtree(boundary,G=1,theta_=thetas[0])
for p in points:
    qt.insert(p)
ax[0,0].scatter([p.x for p in points],[p.y for p in points],s=1)
qt.show_from_point(blackhole_point,axis=ax[0,0])

ax[0,1].set_title(f"Theta = {thetas[1]}")
qt = astrojigs.Quadtree(boundary,G=1,theta_=thetas[1])
for p in points:
    qt.insert(p)
ax[0,1].scatter([p.x for p in points],[p.y for p in points],s=1)
qt.show_from_point(blackhole_point,axis=ax[0,1])

ax[1,0].set_title(f"Theta = {thetas[2]}")
qt = astrojigs.Quadtree(boundary,G=1,theta_=thetas[2])
for p in points:
    qt.insert(p)
ax[1,0].scatter([p.x for p in points],[p.y for p in points],s=1)
qt.show_from_point(blackhole_point,axis=ax[1,0])

ax[1,1].set_title(f"Theta = {thetas[3]}")
qt = astrojigs.Quadtree(boundary,G=1,theta_=thetas[3])
for p in points:
    qt.insert(p)
ax[1,1].scatter([p.x for p in points],[p.y for p in points],s=1)
qt.show_from_point(blackhole_point,axis=ax[1,1])

plt.show()
