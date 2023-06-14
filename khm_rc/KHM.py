import itertools
import logging

import numpy as np


class KHM:
    def __init__(self, function_basis=None, p=2, lib="numpy"):
        """
        :param function_basis: The basis functions for each cluster
        :param p:
        :param lib: 'numpy' for CPU calculations, 'cupy' for CUDA calculations
        """

        self.function_basis = function_basis
        self.K = len(function_basis)
        self.p = p
        self.coeff = None
        self.best_trial = -1
        self.loss = float("inf")
        if lib == "numpy":
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
            s += ", ".join([str(c_) for c_ in c]) + "\n\n\n"

        return s

    def create_X(self, x):
        return [
            self.lib.array(
                [f(x_i) for x_i, f in itertools.product(x, self.function_basis[k])]
            ).reshape((-1, len(self.function_basis[k])))
            for k in range(self.K)
        ]

    def fit(
        self,
        x,
        y,
        max_iterations=100,
        verbose="iteration",
        print_interval=2,
        trials=1,
        eps=1e-3,
        weights=None,
        solver="ridge",
    ):
        """
        :param x:
        :param y:
        :param num_iterations:
        :param verbose: silent - silent
                                        iteration - one line per iteration
                                        end - print only start and end iterations
        :param trials: number of random starts. The algorithm will choose the trial with minimum loss
        :return:
        """
        if len(x) == 0:
            raise Exception("KHM detected empty dataset")
        x = self.lib.array(x)
        y = self.lib.array(y)
        W = self.lib.eye(len(y)) if weights is None else self.lib.diag(weights)
        X = self.create_X(x)
        if verbose == "silent":
            verbose_print = lambda **a: None
        if verbose == "iteration" or verbose == "end":

            def verbose_print(trial, iteration, loss):
                if iteration % print_interval == 0 or iteration == 0:
                    print(
                        f"Trial {trial}	:	Iteration {iteration:{len(str(max_iterations))}d} / {max_iterations}" + 
                        f"		loss : {loss}"
                    )

        if verbose == "end":
            print_interval = max_iterations - 1

        coeff = None
        for t in range(trials):
            cur_loss, cur_coeff = self.run_trial(
                x, X, y, verbose_print, max_iterations, eps, t, W, solver
            )
            if cur_loss < self.loss:
                coeff = cur_coeff
                self.best_trial = t
                self.loss = cur_loss

        self.coeff = coeff

        return cur_loss, cur_coeff

    def run_trial(
        self, x, X, y, verbose_print, max_iterations, eps, trial_num, W, solver
    ):
        coeff = self.step1()
        loss = float("inf")

        for r in range(1, max_iterations + 1):
            d = self.step2(x, y, coeff)
            new_loss = self.calc_loss(d, W)
            verbose_print(iteration=r, loss=new_loss, trial=trial_num)
            if abs(new_loss - loss) < eps:
                break

            loss = new_loss
            if solver == "linear":
                coeff = [
                    self.LinReg_KHM(d=d, k=k, X=X[k], y=y, p=self.p, W=W)
                    for k in range(self.K)
                ]
            elif solver == "ridge":
                coeff = [
                    self.LinReg_ridge_KHM(d=d, k=k, X=X[k], y=y, p=self.p, W=W)
                    for k in range(self.K)
                ]
            else:
                raise NotImplementedError(f"Solver {solver} not implemented")

        d = self.step2(x, y, coeff)
        new_loss = self.calc_loss(d, W)
        verbose_print(iteration=max_iterations - 1, loss=new_loss, trial=trial_num)

        return loss, coeff

    def LinReg_KHM(self, d, k, X, y, p, W, q=2):
        denominator = (
            ((d[:, k]) ** (p + q)) * (1 / (d**p)).sum(axis=1) ** 2
        ).reshape(-1, 1)
        W = self.lib.sqrt(W) / denominator
        if all(y == 0):
            _, s, vh = self.lib.linalg.svd(W @ X / denominator, full_matrices=True)
            v = vh.T
            return v[:, -1]
        else:
            try:
                return self.lib.linalg.inv(X.T @ W @ X) @ X.T @ W.T @ (y.reshape(-1, 1))
            except self.lib.linalg.LinAlgError as e:
                return self.lib.random.randn(len(self.function_basis[k])).reshape(
                    (-1, 1)
                )

    def LinReg_ridge_KHM(self, d, k, X, y, p, W, q=2, lambda_=1e-2):
        denominator = (
            ((d[:, k]) ** (p + q)) * (1 / (d**p)).sum(axis=1) ** 2
        ).reshape(-1, 1)
        W = self.lib.sqrt(W) / denominator
        if all(y == 0):
            _, s, vh = self.lib.linalg.svd(W @ X / denominator, full_matrices=True)
            v = vh.T
            return v[:, -1]
        else:
            try:
                X_T_W = X.T @ W
                return (
                    self.lib.linalg.inv(X_T_W @ X + lambda_ * np.eye(X.shape[1]))
                    @ X_T_W
                    @ (y.reshape(-1, 1))
                )
            except self.lib.linalg.LinAlgError as e:
                return self.lib.random.randn(len(self.function_basis[k])).reshape(
                    (-1, 1)
                )

    def final(self, d, k, X, y, p, W, q=2):
        (relevant_idx,) = self.lib.where(d.argmin(axis=1) == k)
        W = self.lib.diag(self.lib.diag(W)[relevant_idx])
        X = X[relevant_idx]
        y = y[relevant_idx]

        return self.lib.linalg.inv(X.T @ W @ X) @ X.T @ W.T @ (y.reshape(-1, 1))

    def p_values(self, d, p, q=2):
        """Calculates P(Z_k | z_i) given d_{i,j}"""
        d = d ** (p + q)
        p = self.lib.array([[d_i_k for d_i_k in d_i] for d_i in d])

        return p / d.sum(axis=1).reshape(-1, 1)

    def a_values(self, d, p, q=2):
        """Calculates a_p(z_i) given d_{i,j}"""
        return (d ** (p + q)).sum(axis=1) / ((d**p).sum(axis=1)) ** 2

    def calc_kth_function(self, k, x, coeff=None):
        if coeff is None:
            coeff = self.coeff

        res = []
        for xi in x:
            res.append(self.lib.array([f(xi) for f in self.function_basis[k]]))
        res = self.lib.array(res)
        try:
            return (res @ coeff[k].reshape(-1, 1)).flatten()
        except ValueError as e:
            logging.exception(e)
            logging.info("X:")
            logging.info(x)
            logging.info("res:")
            logging.info(res)
            logging.info("coeff:")
            logging.info(coeff[k])

    def step2(self, x, y, coeff=None):
        d = self.lib.vstack(
            [
                self.lib.abs(self.calc_kth_function(k=k, x=x, coeff=coeff) - y)
                for k in range(self.K)
            ]
        ).T
        return d

    def step1(self):
        return [self.lib.random.randn(len(base)) for base in self.function_basis]

    def calc_loss(self, d, W):
        return (self.K / (1 / d).sum(axis=1) @ W.T).sum() / d.shape[0]

    def calc_kmeans_loss(self, x, y, coeff, w):
        d = self.step2(x, y, coeff)
        loss = 0
        pos = d.argmin(axis=1)
        for k in range(self.K):
            r_x = x[np.where(pos == k)]
            r_y = y[np.where(pos == k)]
            r_w = w[np.where(pos == k)]
            W = np.diag(r_w)

            if len(r_x) == 0:
                continue

            loss_tmp = (
                self.lib.array(
                    self.lib.abs(self.calc_kth_function(k, r_x, coeff) - r_y)
                )
                @ W
            )
            loss += loss_tmp.sum()

        return loss

    def get_best_functions(self, x, y, w):
        """Finds the best among the k functions that fit the given data."""
        d = self.step2(x, y)
        pos = d.argmin(axis=1)
        losses = np.zeros(self.K)
        counts = np.zeros(self.K)
        for k in range(self.K):
            r_x = x[np.where(pos == k)]
            r_y = y[np.where(pos == k)]
            r_w = w[np.where(pos == k)]
            W = np.diag(r_w)

            if len(r_x) == 0:
                losses[k] = 0
                counts[k] = 0
                continue

            losses[k] = (
                self.lib.array(self.lib.abs(self.calc_kth_function(k, r_x) - r_y)) @ W
            ).mean()
            counts[k] = r_x.size

        return losses, counts
