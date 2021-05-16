import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class AnimatedScatter(object):
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation
        self.ani = animation.FuncAnimation(self.fig, self.update, interval = 5,
                                        init_func=self.setup_plot, blit=True)


#Function to give the next positions:
def step(mass:'arr', x0:'arr', y0:'arr', vx0:'arr', vy0:'arr',dt=0.1,G=1):

    #updating the positions
    x1 = x0 + vx0*dt
    y1 = y0 + vy0*dt

    vx1=[]
    vy1=[]
    # Loop over planets to find the distance:
    for i in range(len(x1)):
        x1self = x1[i]
        y1self = y1[i]

        ax=0
        ay=0

        for i in range(len(x0)):
            # if -statement for avoiding same body calculation
            if i==j:
                continue
            x_dist = x1[j] - x1self
            y_dist = y1[j] - y1self
            Rsq = x_dist**2 + y_dist**2

            # Contribution to acceleration of ith mass by jth mass
            a = G * mass[j]/Rsq
            ax += a * x_dist/np.sqrt(Rsq)
            ay += a * y_dist/np.sqrt(Rsq)
        vx1.append(vx0[i] + ax*dt)
        vy1.append(vy0[i] + ay*dt)

    return x1, y1, np.array(vx1), np.array(vy1)

def scale_the_array(arr, min_=7, max_=300):
    l = []
    for element in arr:
        if arr.max() != arr.min():
            scaled_element = ((element - arr.min())/(arr.max()-arr.min()))*(max_-min_) + min_
            l.append(scaled_element)
        else:
            # Default size
            l = [20, 20]

    return np.array(l)
