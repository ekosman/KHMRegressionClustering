import warnings

from scipy.ndimage import binary_opening, binary_dilation, binary_erosion
from scipy.ndimage.filters import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.exceptions import ConvergenceWarning

from KHM import KHM
import pandas as pd


def binarize(data):
	model = mixture.GaussianMixture(n_components=6)
	model.fit(data.flatten().reshape(-1, 1))
	preds = model.predict(data.flatten().reshape(-1, 1))
	low_ = np.argmin(model.means_)

	preds = (preds != low_).astype(np.int)

	data = preds.reshape(data.shape)

	data = binary_opening(data, structure=np.ones((3, 3)), iterations=2)
	data = binary_erosion(data, structure=np.ones((3, 3)), iterations=2, border_value=1)
	data = data.astype(int)

	return data


def binarize_grad(grad):
	model = mixture.GaussianMixture(n_components=2)
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			model.fit(grad.flatten().reshape(-1, 1))
		except ConvergenceWarning as e:
			print(e)
			exit()
	preds = model.predict(grad.flatten().reshape(-1, 1))
	low_ = np.argmin(model.means_)

	preds = (preds != low_).astype(np.int)

	plt.figure()
	for c in range(len(np.unique(preds))):
		idx, = np.where(preds == c)
		plt.scatter(grad.flatten()[idx], [0]*len(idx))
	plt.show()

	grad_ga_binary = preds.reshape(grad.shape)
	return grad_ga_binary


def make_grid(df):
	points = df.values[..., :-1].astype(int)
	points = np.roll(points, 1)
	mins = points.min(axis=0)
	values = df.values[:, -1]

	grid = np.zeros([max(v) - min(v) + 1 for v in points.T])
	grid[tuple([col for col in (points - mins).T])] = values
	return grid


def prepare_data(df):
	grid = make_grid(df)
	plt.matshow(grid, interpolation='nearest', origin='lower')
	plt.show()
	grid = gaussian_filter(grid, sigma=3)

	# grid_bin = binarize(grid)

	grad = np.linalg.norm(np.stack(np.gradient(grid)), ord=2, axis=0)

	plt.matshow(grad, interpolation='nearest', origin='lower')
	plt.show()

	grad = binarize_grad(grad)

	plt.matshow(grad, interpolation='nearest', origin='lower')
	plt.show()

	idxs = np.array([col.tolist() for col in np.where(grad == grad)])
	x = idxs[1:, :].T
	y = idxs[0, :]
	weights = grad[tuple([col for col in idxs])]
	relevant, = np.where(weights > 0)
	x = x[relevant]
	y = y[relevant]
	weights = weights[relevant]

	return x, y, weights, grid, grad


def get_coeff(df, bases, plot=False):
	x, y, weights, data, grad = prepare_data(df)

	model = KHM(function_basis=bases)
	cur_loss, cur_coeff = model.fit(x=x, y=y, max_iterations=100, trials=20, verbose='end', weights=weights, eps=1e-4)
	print(repr(model))
	if plot:
		fig = plt.figure()

		ax = fig.add_subplot(151)
		plt.title("Gradient with lines")

		cax = ax.matshow(data, interpolation='nearest', origin='lower')

		x_ = np.array(list(range(data.shape[1]))).reshape(-1, 1)
		for k in range(len(bases)):
			y_ = np.array(model.calc_kth_function(k, x_))
			ax.plot(x_ - 0, y_ - 0)
		plt.xlim(0, data.shape[1])
		plt.ylim(0, data.shape[0])

		ax = fig.add_subplot(152)
		cax = ax.matshow(grad, interpolation='nearest', origin='lower')

		ax = fig.add_subplot(153)
		fun_losses, counts = model.get_best_functions(x, y, weights)
		for i, l in enumerate(fun_losses):
			ax.scatter([i], [l])
			ax.text(i, l, str(counts[i]))

		ax = fig.add_subplot(154)
		ax.hist(counts, bins=100)

		ax = fig.add_subplot(155)
		cax = ax.matshow(data, interpolation='nearest', origin='lower')
		chosen = np.argsort(-counts)[:5]
		chosen_loss = [fun_losses[i] for i in chosen]
		chosen_loss_indexes = np.argsort(chosen_loss)[:3]
		chosen = [chosen[i] for i in chosen_loss_indexes]
		for k in chosen:
			y_ = np.array(model.calc_kth_function(k, x_))
			ax.plot(x_ - 0, y_ - 0)
		plt.show()

	return cur_loss, cur_coeff, model.calc_kmeans_loss(x=x, y=y, coeff=model.coeff, w=weights)


def get_coeff_trials(df, base, num_trials):
	losses = []
	coeffs = []
	# for i in range(num_trials):
	# 	print(f"======== {i} ========")
		# loss, coeff, k_loss = get_coeff(df=df, bases=[base] * (i + 1), plot=True)
		# losses.append(k_loss)
		# coeffs.append(coeff)
	i = 15
	loss, coeff, k_loss = get_coeff(df=df, bases=[base] * (i + 1), plot=True)
	return losses, coeffs


if __name__ == '__main__':
	df = pd.read_csv(r'C:\Users\eitan\Desktop\exps3\3_4_5_6_15\seq_len_20_P[E3.2 ! E4.1].csv')
	relevant_data = df.apply(lambda row: all([e != -1 for e in row]), axis=1)
	df = df[relevant_data]
	base = [lambda x: x[0] ** 2, lambda x: x[0], lambda x: 1]

	losses, coeffs = get_coeff_trials(df=df, base=base, num_trials=10)

	plt.figure()
	plt.plot(losses, label='losses')
	plt.plot(np.gradient(losses), label='gradient')
	plt.plot(np.gradient(np.gradient(losses)), label='hessian')
	plt.legend()
	plt.show()