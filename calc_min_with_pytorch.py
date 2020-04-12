
# 用pytorch算 'y = (5x1 + 3x2 - 1)^2 + (-3x1 - 4x2 + 1)^2' 的最小值

import torch
from torch.autograd import Variable

def calc_min(func, lr=0.0001, threshold=1e-8):
	x1 = torch.randn(1, requires_grad=True)
	x2 = torch.randn(1, requires_grad=True)

	i = 0
	while True:
		out = func(x1, x2)
		out.backward()
		print("item{}: x1: {} x2: {} out: {}".format(i, x1.item(), x2.item(), out.item()))
		# print(x1.grad)  # 68*x1+54*x2-16
		# print(x2.grad)  # 54*x1+50*x2-14
		x1.data -= lr * x1.grad * out.data
		x2.data -= lr * x2.grad * out.data

		out_ = func(x1, x2)

		delta = torch.abs(out_ - out)

		if delta < threshold: break

		i += 1

	return out_.item()

def func(x1, x2):
	return torch.pow(5 * x1 + 3 * x2 - 1, 2) + torch.pow(-3 * x1 - 4 * x2 + 1, 2)

if __name__ == '__main__':
	min_y = calc_min(func)
	print(min_y)