import multiprocessing as mp
import random
import string


def cube(x, y):
	return x, y


if __name__ == '__main__':
	pool = mp.Pool(processes=4)
	results = [pool.apply(cube, args=(x, y)) for x, y in zip(range(1, 7), range(6, 12))]
	print(results)
