import itertools

import numpy as np
import matplotlib.pyplot as plt
from KHM import KHM
from utils import calc_f, register_logger
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
	register_logger()
	coeff1 = [1, 1]
	coeff2 = [-1, 1, -0.5, 500]
	basis1 = [lambda x: x[0] ** 2, lambda x: x[1] ** 2]
	basis2 = [lambda x: x[0] ** 2, lambda x: x[1], lambda x: x[0] * x[1], lambda x: 1]

	x1 = np.mgrid[-50:50:4, -50:50:4].reshape(2, -1).T
	x2 = np.mgrid[-50:50:4, -50:50:4].reshape(2, -1).T
	xn = np.mgrid[-50:50:4, -50:50:4].reshape(2, -1).T

	y1 = np.array([calc_f(basis=basis1, coeff=coeff1, x=x_i) + np.random.randn() * 40 for x_i in x1])
	y2 = np.array([calc_f(basis=basis2, coeff=coeff2, x=x_i) + np.random.randn() * 40 for x_i in x2])
	yn = np.array([np.random.uniform(low=np.concatenate([y1, y2]).min(),
									 high=np.concatenate([y1, y2]).max(),
									 size=len(xn))]).flatten()

	x = np.concatenate([x1, x2, xn])
	y = np.concatenate([y1, y2, yn])

	weights = [1] * (len(y1) + len(y2)) + [0.2] * len(yn)

	model = KHM(function_basis=[basis1, basis2])
	model.fit(x=x, y=y, max_iterations=50, trials=5, verbose='iteration', eps=1e-3, print_interval=1, weights=weights)
	print(repr(model))

	x_1 = np.arange(-50, 50, 4)
	x_2 = np.arange(-50, 50, 4)
	x = [[b, a] for a, b in itertools.product(x_1, x_2)]
	res1 = model.calc_kth_function(k=0, x=x).reshape(len(x_1), len(x_2))
	res2 = model.calc_kth_function(k=1, x=x).reshape(len(x_1), len(x_2))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter([a[0] for a in x1], [a[1] for a in x1], y1, s=20, label='f1')
	ax.scatter([a[0] for a in x2], [a[1] for a in x2], y2, s=20, label='f2')
	ax.scatter([a[0] for a in xn], [a[1] for a in xn], yn, s=5, label='noise')

	X, Y = np.meshgrid(x_1, x_2)
	ax.plot_surface(X, Y, res1, linewidth=0)
	ax.plot_surface(X, Y, res2, linewidth=0)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.legend()
	plt.show()
