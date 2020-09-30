"""
	This program shows visualizations for our dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
from functions import prepare_data, get_weights

def main() -> None:
	"""Runs the program, plots raw and normalized dataset. If the weights are not null, we draw them as a red line."""
	data = pd.read_csv("./data.csv")
	_, ax = plt.subplots(1,2)
	ax[0].scatter(data=data, x="km", y="price")
	ax[0].set_title("Raw dataset")
	theta0, theta1 = get_weights()
	if (theta0 != 0 and theta1 != 0):
		x = data["km"]
		y = theta0 + theta1 * x
		ax[0].plot(x, y, 'r')
	prepare_data(data)
	ax[1].set_title("Normalized dataset")
	ax[1].scatter(data=data, x="km", y="price")
	plt.show()

if (__name__ == "__main__"):
	main()