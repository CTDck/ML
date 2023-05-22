"""Generate various different attractors"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output


class Attractor:
    """Create attractor object"""

    def __init__(self, step_size, iteration):
        self.step_size = step_size
        self.iter = iteration
        self.fig = plt.figure()
        self.ax = plt.axes(projection="3d")

    def lorenz_setup(self, xyz, *, s=10, r=28, b=2.667):
        """Setup lorenz set of differential equations

        Args:
            xyz (array): array of x, y and z values
            s (int, optional): tweaking param 1. Defaults to 10.
            r (int, optional): tweaking param 2. Defaults to 28.
            b (float, optional): tweaking param 3. Defaults to 2.667.

        Returns:
            array: dx/dt, dy/dt, dz/dt
        """
        x, y, z = xyz
        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z

        return np.array([x_dot, y_dot, z_dot])

    def newton_leipnik(self, xyz, *, a=0.4, b=0.175):
        """Setup Newton Leipnik set of differential equations

        Args:
            xyz (_type_): array of x, y and z values
            a (float, optional): tweaking parameter 1. Defaults to 0.4.
            b (float, optional): tweaking parameter 2. Defaults to 0.175.

        Returns:
            array: dx/dt, dy/dt, dz/dt
        """
        x, y, z = xyz
        x_dot = -a * x + y + 10 * y * z
        y_dot = -x - 0.4 * y + 5 * x * z
        z_dot = b * z - 5 * x * y

        return np.array([x_dot, y_dot, z_dot])

    def create_points(self, starting_values, method):
        """Create array of 3d points to plot the attractor

        Args:
            starting_values (tuple): 3d starting point
            method (function): attractor to transform the point

        Returns:
            array: all iterations of the transformed point along [0, iteration]
        """
        points = np.empty((self.iter + 1, 3))
        points[0] = starting_values
        for i in range(self.iter):
            points[i + 1] = points[i] + method(points[i]) * self.step_size

        return points

    def init_figure(self):
        """Initialise plotting"""
        self.ax.clear()
        plt.axis("off")
        self.ax.grid(False)

    def update_figure(self, t, name, points_group):
        """Update the plotted figure with appropriate name, frame instance and passes the array of points

        Args:
            t (int): current frame number
            name (string): Name of the Attractor
            points (array): array of points to plot
        """
        self.ax.clear()
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.axis("off")
        if t < 100:
            self.ax.set_title(f"{name}", alpha=(t / 100), fontsize=20)
        else:
            self.ax.set_title(f"{name}", alpha=1, fontsize=20)
        self.ax.plot3D(*points_group[0].T, color="orange", lw=0.5, alpha=0.8)
        for points in points_group:
            self.ax.scatter(*points[t].T, marker="o", color="red", s=5)
            if t > 25:
                self.ax.plot3D(*points[t - 25 : t].T, color="red", lw=0.5, alpha=1)
            else:
                self.ax.plot3D(*points[:t].T, color="red", lw=0.5, alpha=1)
        self.ax.view_init(30, t / 1000 * 180)
        clear_output(wait=True)
        print(f"Iteration No: {t} out of {self.iter}")

    def draw_figure(self, name, points):
        """Draw the figure

        Args:
            name (string): Name of the attractor to draw
            points (array): array of points to plot
        """
        self.fig.set_size_inches(10, 10)
        self.ax.grid(False)
        plt.axis("off")

        anim = FuncAnimation(
            self.fig,
            self.update_figure,
            frames=np.arange(0, self.iter, 1),
            init_func=self.init_figure,
            fargs=(
                f"{name}",
                points,
            ),
            interval=1,
        )
        anim.save(f"Data/{name}.mp4", dpi=150, fps=30, writer="ffmpeg")