import torch.nn as nn
import torch


class Base(nn.Module):
	def __init__(self, base):
		super(Base, self).__init__()
		self.base = base

	def forward(self, x):
		res = []
		for xi in x:
			res.append(torch.tensor([f(xi) for f in self.base]))

		return torch.stack(res)


class Model(nn.Module):
	def __init__(self, base):
		super(Model, self).__init__()
		self.base = Base(base)
		# self.coeff = nn.Parameter(torch.randn(len(base)), requires_grad=True)
		self.fc = nn.Linear(len(base), 1, bias=False)

	def forward(self, x):
		x = x.float()
		x = self.base(x)
		x = self.fc(x)
		return x


class RegressionLoss(nn.Module):
	def __init__(self):
		super(RegressionLoss, self).__init__()

	def forward(self, a, p, y_, y, q, w):
		return (w * a * p * torch.abs(y_ - y) ** q).mean()
