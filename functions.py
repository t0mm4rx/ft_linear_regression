"""
	This file contains useful functions used globally.
"""

import numpy as np
import pandas as pd
from typing import Tuple

def prepare_data(data: pd.DataFrame) -> Tuple[float, float]:
	"""Normalize data."""
	max_km = data["km"].max()
	max_price = data["price"].max()
	# min_km = data["km"].min()
	# min_price = data["price"].min()
	# data["km"] = (data["km"] - min_km) / (max_km - min_km)
	# data["price"] = (data["price"] - min_price) / (max_price - min_price)
	data["km"] = data["km"] / max_km
	data["price"] = data["price"] / max_price
	# return (min_km, max_km, min_price, max_price)
	return (max_km, max_price)

def predict(theta0: float, theta1: float, kilometrage: float) -> float:
	"""Returns an estimated price prediction for given weights and kilometrage."""
	return theta0 + theta1 * kilometrage

def get_weights() -> Tuple[float, float]:
	"""Get weights from the disk, if the weights don't exist, we create the file with 0, 0."""
	try:
		thetas = np.load("weights.npy")
		return (thetas[0], thetas[1])
	except:
		save_weights(0, 0)	
	return (0, 0)

def save_weights(theta0: float, theta1: float) -> None:
	"""Save weights to disk."""
	np.save("weights", np.array([theta0, theta1]))
	print("Weigths saved on this disk.")