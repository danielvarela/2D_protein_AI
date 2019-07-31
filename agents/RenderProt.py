import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

class RenderProt():
    def __init__(self, seq):
        self.seq = seq
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.init_axis()
        plt.ion()
        plt.show()

    def run(self):
        center = 0
        while True:
            points = [(0,np.random.randint(-1,1),i) for i in range(0, len(self.seq))]
            self.render(points, center)
            center = (center + 1) % len(self.seq)
            time.sleep(10)
        
    def init_axis(self):
        self.ax.set_xlim3d(-5,5)
        self.ax.set_ylim3d(-5,5)
        self.ax.set_zlim3d(-5,5)
 
    def plot_sphere(self, center):
        r = 1.5
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u),np.sin(v)) + center[0]
        y = r * np.outer(np.sin(u),np.sin(v)) + center[1]
        z = r * np.outer(np.ones(np.size(u)),np.cos(v)) + center[2]
        sphere = self.ax.plot_wireframe(x,y,z,color='r',alpha=0.1) 
        
    def render(self, points, center):
        self.ax.cla()
        self.init_axis()
        x = [x for x,y,z in points]
        y = [y for x,y,z in points]
        z = [z for x,y,z in points]
        h_x = [x for i,(x,y,z) in enumerate(points) if self.seq[i] == 'H' ]
        h_y = [y for i,(x,y,z) in enumerate(points) if self.seq[i] == 'H' ]
        h_z = [z for i,(x,y,z) in enumerate(points) if self.seq[i] == 'H' ]
        self.ax.plot(x, y, z, linestyle='-', marker="o", color="b")
        self.ax.plot(h_x, h_y, h_z, linestyle='None', marker="o", color="w")
        self.plot_sphere(points[center])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
