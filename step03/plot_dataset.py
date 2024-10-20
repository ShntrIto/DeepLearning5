import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), 'height_weight.txt')
xs = np.loadtxt(path)

print('xs.shape: ', xs.shape)
xs = xs[:500]

# small_xs = xs[:500]
# plt.scatter(small_xs[:, 0], small_xs[:, 1])
# plt.xlabel('Height(cm)')
# plt.ylabel('Weight(kg)')
# plt.savefig('output/plot_dataset.png')

import numpy as np
import matplotlib.pyplot as plt

def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y

mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False)

grid_size = 0.1

x = np.arange(xs[:, 0].min()-grid_size, xs[:, 0].max()+grid_size, grid_size)
y = np.arange(xs[:, 1].min()-grid_size, xs[:, 1].max()+grid_size, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)
        
fig = plt.figure()
fig.subplots_adjust(wspace=0.5)  # Add more space between the plots

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('Height(cm)')
ax2.set_ylabel('Weight(kg)')
ax2.contour(X, Y, Z)
ax2.scatter(xs[:, 0], xs[:, 1])

plt.savefig('output/plot_dataset_3d_small.png')