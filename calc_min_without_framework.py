
# 求 'y=x^2' 的最小值，使用梯度下降算法，不用框架

import random
import matplotlib.pyplot as plt
from sympy import diff, symbols, solve


def calc_min(vals, func, lr=0.1, threshold=1e-5):
	symbol = [symbols("x{}".format(i)) for i in range(vals)]
	y = func(*symbol)

	xs = [random.randint(-10, 10) for _ in range(vals)]

	while True:
		delta = 0.
		for i, (s, x) in enumerate(zip(symbol, xs)):
			d_func = diff(y, s)  # 对s的偏导数
			x_ = float(d_func.subs(s, x))  # 代入导数的结果
			x -= lr * x_  # x = x - lr * x_  减学习率乘梯度
			ori_y = y.evalf(subs=dict(zip(symbol, xs)))  # 代入
			xs[i] = x
			new_y = y.evalf(subs=dict(zip(symbol, xs)))
			delta += abs(new_y - ori_y)
			print(new_y)
		if delta < threshold: break

	return y.evalf(subs=dict(zip(symbol, xs)))



if __name__ == '__main__':
	# calc_min(2, lambda x1, x2: pow(x1, 2) + pow(x2, 2) + 3)
	min_y = calc_min(1, lambda x1: pow(x1, 2))