import numpy as np
import matplotlib.pyplot as plt
from KHM import KHM
from utils import calc_f, register_logger

if __name__ == '__main__':
	register_logger()
	coeff1 = [0.2, -0.5, 5]
	coeff2 = [-0.2, -10, -5]
	coeff3 = [0, 4, 0]
	basis1 = [lambda x: x[0]**2, lambda x: x[0], lambda x: 1]
	basis2 = [lambda x: x[0]**2, lambda x: x[0], lambda x: 1]
	basis3 = [lambda x: x[0]**2, lambda x: x[0], lambda x: 1]

	x1 = np.linspace(-50, 50, 100).reshape(-1, 1)
	x2 = np.linspace(-50, 50, 100).reshape(-1, 1)
	x3 = np.linspace(-50, 50, 100).reshape(-1, 1)
	y1 = np.array([calc_f(basis=basis1, coeff=coeff1, x=x_i) + np.random.randn()*20 for x_i in x1])
	y2 = np.array([calc_f(basis=basis2, coeff=coeff2, x=x_i) + np.random.randn()*20 for x_i in x2])
	y3 = np.array([calc_f(basis=basis3, coeff=coeff3, x=x_i) + np.random.randn()*20 for x_i in x3])

	x = np.concatenate([x1, x2, x3])
	y = np.concatenate([y1, y2, y3])

	model = KHM(function_basis=[basis1, basis2, basis3])
	model.fit(x=x, y=y, max_iterations=10, trials=10, verbose='iteration')
	print(repr(model))

	x_1 = np.linspace(-80, 80, 100).reshape(-1, 1)
	x_2 = np.linspace(-80, 80, 100).reshape(-1, 1)
	x_3 = np.linspace(-80, 80, 100).reshape(-1, 1)
	res1 = model.calc_kth_function(k=0, x=x_1)
	res2 = model.calc_kth_function(k=1, x=x_2)
	res3 = model.calc_kth_function(k=2, x=x_3)

	plt.figure()
	plt.scatter(x1, y1, label='f1', s=15)
	plt.scatter(x2, y2, label='f2', s=15)
	plt.scatter(x3, y3, label='f3', s=15)
	plt.plot(x_1, res1)
	plt.plot(x_2, res2)
	plt.plot(x_3, res3)
	plt.grid()
	plt.legend()
	plt.show()