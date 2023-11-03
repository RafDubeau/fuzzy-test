import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


points1 = np.random.rand(10, 2)
points2 = np.random.rand(14, 2)
points3 = np.random.rand(12, 2)

points = [points1, points2, points3]


fig, ax = plt.subplots()


scat = ax.scatter(points[0][:, 0], points[0][:, 1], c="b", s=5)
ax.set(xlim=[0, 1], ylim=[0, 1])


def update(frame):
    data = points[frame]

    # update the scatter plot:
    scat.set_offsets(data)

    return scat


ani = animation.FuncAnimation(fig=fig, func=update, frames=len(points), interval=500)
plt.show()