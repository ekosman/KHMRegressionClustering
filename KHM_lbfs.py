import itertools
import logging
import torch

import numpy as np

from model import Model, RegressionLoss


class KHM:
	def __init__(self, function_basis=None, p=2, lib='numpy'):
		"""
		:param function_basis: The basis functions for each cluster
		:param p:
		:param lib: 'numpy' for CPU calculations, 'cupy' for CUDA calculations
		"""

		self.function_basis = function_basis
		self.K = len(function_basis)
		self.p = p
		self.models = [Model(base) for base in function_basis]
		# self.optimizers = [torch.optim.LBFGS(params=model.parameters(),
		# 									 line_search_fn='strong_wolfe') for model in self.models]
		self.optimizers = [torch.optim.SGD(params=model.parameters(),
											 lr=1e-7) for model in self.models]
		self.loss_fn = None
		self.best_trial = -1
		self.loss = float("inf")
		if lib == 'numpy':
			import numpy as np
		else:
			import cupy as np
		self.lib = np

	def __repr__(self):
		s = f"""
loss: {self.loss}
best trial: {self.best_trial}

====== Coefficients ======
		"""
		for c in self.coeff:
			s += ', '.join([str(c_) for c_ in c]) + '\n\n\n'

		return s

	def create_X(self, x):
		return [self.lib.array([f(x_i) for x_i, f in itertools.product(x, self.function_basis[k])]).reshape(
			(-1, len(self.function_basis[k]))) for k in range(self.K)]

	def fit(self, x, y, max_iterations=100, verbose=0, print_interval=2, trials=1, eps=1e-3, weights=None, q=2):
		"""
		:param x:
		:param y:
		:param num_iterations:
		:param verbose: 0 - silent
						1 - one line per iteration
						2 - print only start and end iterations
		:param trials: number of random starts. The algorithm will choose the trial with minimum loss
		:return:
		"""
		self.loss_fn = [RegressionLoss() for i in range(self.K)]
		x = torch.tensor(x)
		y = torch.tensor(y)
		if weights is None:
			weights = [1] * len(y)
		weights = torch.tensor(weights)
		if verbose == 0:
			verbose_print = lambda **a: None
		if verbose == 1 or verbose == 2:
			def verbose_print(trial, iteration, loss):
				if iteration % print_interval == 0 or iteration == 0:
					print(f"Trial {trial}	:	Iteration {iteration: {len(str(max_iterations))}d} / {max_iterations}		loss : {loss}")
		if verbose == 2:
			print_interval = max_iterations - 1

		for t in range(trials):
			cur_loss = self.run_trial(x, y, verbose_print, max_iterations, eps, t, weights)
			# cur_loss, cur_coeff = self.run_trial(x, y, verbose_print, max_iterations, eps, t, weights)
			if cur_loss < self.loss:
				# coeff = cur_coeff
				self.best_trial = t
				self.loss = cur_loss

		# self.coeff = coeff

		return cur_loss
		# return cur_loss, cur_coeff

	def run_trial(self, x, y, verbose_print, max_iterations, eps, trial_num, w):
		# loss = self.calc_loss(x, y, coeff, W)
		# verbose_print(iteration=0, loss=loss, trial=trial_num)
		loss = float("inf")
		for r in range(1, max_iterations + 1):
			new_loss = 0
			d = self.step2(x, y)
			a_values = self.a_values(d, self.p)
			p_values = self.p_values(d, self.p)
			for k in range(self.K):
				new_loss += self.lbfs_KHM(k=k, a_values=a_values, p_values=p_values, x=x, y=y, w=w)
			# new_loss = self.calc_loss(x, y, coeff, W)
			verbose_print(iteration=r, loss=new_loss, trial=trial_num)

			if abs(new_loss - loss) < eps:
				verbose_print(iteration=max_iterations - 1, loss=new_loss, trial=trial_num)
				loss = new_loss
				break

			loss = new_loss

		return loss

	def lbfs_KHM(self, k, a_values, p_values, x, y, w, q=2):
		loss_item = 0
		# print(self.models[k].coeff)

		def closure():
			nonlocal loss_item
			self.optimizers[k].zero_grad()
			output = self.models[k](x)
			loss = self.loss_fn[k](a_values, p_values[:, k], output, y, q, w)
			loss.backward()

			loss_item = loss.item()
			return loss

		self.optimizers[k].step(closure)
		# print(self.models[k].coeff)
		return loss_item

	def p_values(self, d, p, q=2):
		"""
		Calculates P(Z_k | z_i) given d_{i,j}
		"""
		d = d ** (p + q)

		return d / d.sum(dim=1).reshape(-1, 1)

	def a_values(self, d, p, q=2):
		"""
		Calculates a_p(z_i) given d_{i,j}
		"""
		return (d ** (p + q)).sum(dim=1) / ((d ** p).sum(dim=1)) ** 2

	def step2(self, x, y):
		x = torch.tensor(x)
		d = (torch.stack([model(x).reshape(-1) for model in self.models]) - y).transpose(0, 1).abs().detach()
		return d + 1e-7

	def step1(self):
		return [self.lib.random.randn(len(base)) for base in self.function_basis]

	def calc_loss(self, x, y, coeff, W):
		residuals = self.lib.array([self.lib.linalg.norm([self.calc_kth_function(k, x_i, coeff) - y_i], ord=self.p)
									for (x_i, y_i), k in itertools.product(zip(x, y), range(self.K))]) \
			.reshape(-1, self.K)

		return (self.K / (1 / residuals).sum(axis=1) @ W.T).sum() / len(y)

	def calc_kmeans_loss(self, x, y, coeff, d, W):
		loss = 0
		pos = d.argmin(axis=1)
		w = self.lib.diag(W)
		for k in range(self.K):
			r_x = x[np.where(pos == k)]
			r_y = y[np.where(pos == k)]
			r_w = w[np.where(pos == k)]
			W = np.diag(r_w)

			loss_tmp = self.lib.array([self.lib.linalg.norm([self.calc_kth_function(k, x_i, coeff) - y_i], ord=self.p)
									   for x_i, y_i in zip(r_x, r_y)]) @ W
			loss += loss_tmp.sum()

		print(loss)
		return loss
