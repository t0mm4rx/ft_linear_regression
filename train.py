"""
	This program finds best theta0 and theta1 to best fit the dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import prepare_data, predict, save_weights
from typing import Tuple

def error(prediction: np.ndarray, target: np.ndarray) -> float:
	"""Returns the error between our prediction and the actual price."""
	return prediction - target

def train(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float, learning_rate: float) -> Tuple[float, float]:
	"""Train the weights with given features and targets, returns updated weights."""
	predictions = predict(theta0, theta1, features)
	errors = error(predictions, targets)
	delta0 = learning_rate * (1 / errors.shape[0]) * np.sum(errors)
	delta1 = learning_rate * (1 / errors.shape[0]) * np.sum(errors * features)
	return (theta0 - delta0, theta1 - delta1)

def cost(features: np.ndarray, targets: np.ndarray, theta0: float, theta1: float) -> float:
	"""Return the average error for given weights."""
	predictions = predict(theta0, theta1, features)
	errors = np.abs(error(predictions, targets))
	return (1 / errors.shape[0]) * np.sum(errors)

def main() -> None:
	"""Run the training and save the weights."""
	data = pd.read_csv("./data.csv")
	max_km, max_price = prepare_data(data)
	data = data.values
	features = data[:,0]
	targets = data[:,1]
	learning_rate = 0.01
	epochs = 1000
	batch_size = 4
	errors = []
	theta0, theta1 = (0.0, 0.0)
	for epoch in range(1, epochs + 1):
		for b in range(0, data.shape[0], batch_size):
			theta0, theta1 = train(features[b:b + batch_size], targets[b:b + batch_size], theta0, theta1, learning_rate)
		avg_error = cost(data[:, 0], data[:, 1], theta0, theta1)
		errors.append(avg_error)
		print("Epoch {:4}/{:4}, average error: {:.6f}".format(epoch, epochs, avg_error))
	plt.plot(np.array(errors))
	plt.show()
	theta0 *= max_price
	theta1 *= (max_price / max_km)
	print("Theta0: {:.4f}".format(theta0))
	print("Theta1: {:.4f}".format(theta1))
	save_weights(theta0, theta1)

if (__name__ == "__main__"):
	main()
