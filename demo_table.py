import itertools
from scipy.ndimage.filters import *
import numpy as np
import matplotlib.pyplot as plt
from KHM import KHM
import pandas as pd


def get_coeff(df, bases, plot=False):
	columns = df.columns
	df = df.loc[(df[columns[0]] != -1) & (df[columns[1]] != -1)]

	x_data = df[columns[0]]  # v
	y_data = df[columns[1]]  # u

	min_x = x_data.min()
	max_x = x_data.max()

	min_y = y_data.min()
	max_y = y_data.max()

	data = np.zeros((max_y - min_y + 1, max_x - min_x + 1))
	data[y_data - min_y, x_data - min_x] = df[columns[2]]

	grad_ga = np.linalg.norm(np.stack(np.gradient(gaussian_filter(data, sigma=4))), ord=2, axis=0)

	threshold = np.mean(grad_ga)
	grad_ga = grad_ga >= threshold

	idxs = np.array([col.tolist() for col in np.where(grad_ga==grad_ga)])
	x = idxs[1:, :].T
	y = idxs[0, :]
	weights = grad_ga[tuple([col for col in idxs])]
	relevant, = np.where(weights > 0)
	x = x[relevant]
	y = y[relevant]
	weights = weights[relevant]
	model = KHM(function_basis=bases)
	cur_loss, cur_coeff = model.fit(x=x, y=y, max_iterations=100, trials=1, verbose=2, weights=weights, eps=1e-4)
	print(repr(model))
	if plot:
		grad = np.linalg.norm(np.stack(np.gradient(data)), ord=2, axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(131)
		plt.title("data")
		cax = ax.matshow(data, interpolation='nearest', origin='lower')

		ax = fig.add_subplot(132)
		plt.title("Gradient")

		cax = ax.matshow(grad, interpolation='nearest', origin='lower')
		# fig.colorbar(cax)

		ax = fig.add_subplot(133)
		plt.title("Gradient with lines")

		cax = ax.matshow(grad, interpolation='nearest', origin='lower')
		# fig.colorbar(cax)
		# x = np.array(list(range(0, 32)))
		x = np.array(list(range(grad.shape[1]))).reshape(-1, 1)
		y = np.array(model.calc_kth_function(0, x))
		ax.plot(x - min_x, y - min_y)
		# x = np.array(list(range(grad.shape[1])))
		# y = np.array(model.calc_kth_function(1, x))
		# ax.plot(x - min_x, y - min_y)
		plt.show()

	return cur_loss, cur_coeff, model.calc_kmeans_loss(x=x, y=y, coeff=model.coeff, w=weights)


def get_coeff_trials(df, base, num_trials):
	losses = []
	coeffs = []
	for i in range(num_trials):
		# print(f"======== {i} ========")
		loss, coeff, k_loss = get_coeff(df=df, bases=[base] * (i + 1), plot=False)
		losses.append(k_loss)
		coeffs.append(coeff)

	return losses, coeffs


if __name__ == '__main__':
	df = pd.read_csv(r'seq_len_8_state_P[event0  attr2 ! event0  attr1].csv')
	base = [lambda x: x[0] ** 2, lambda x: x[0], lambda x: 1]

	losses, coeffs = get_coeff_trials(df=df, base=base, num_trials=10)

	plt.figure()
	plt.plot(losses, label='losses')
	plt.plot(np.gradient(losses), label='gradient')
	plt.plot(np.gradient(np.gradient(losses)), label='hessian')
	plt.legend()
	plt.show()