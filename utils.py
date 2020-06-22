import logging
import sys

import numpy as np


def calc_f(basis, coeff, x):
	return np.array(coeff).T @ np.array([f(x) for f in basis])


def register_logger():
	log = logging.getLogger()  # root logger
	for hdlr in log.handlers[:]:  # remove all old handlers
		log.removeHandler(hdlr)

	logging.basicConfig(format="%(asctime)s %(message)s",
						handlers=[
							logging.StreamHandler(stream=sys.stdout)
						],
						level=logging.INFO,
						)
	logging.root.setLevel(logging.INFO)