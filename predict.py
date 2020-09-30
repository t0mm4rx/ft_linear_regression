"""
	This program predicts the price of a car with its kilometrage.
"""

from functions import predict, get_weights

def main() -> None:
	"""Executes the program."""
	kilometrage = input("Kilometrage: ")
	try:
		kilometrage = int(kilometrage)
	except:
		print("Cannot cast '{}' to float.".format(kilometrage))
		exit(1)
	theta0, theta1 = get_weights()
	prediction = predict(theta0, theta1, kilometrage)
	print("Estimated price for {}kms: {:.4f}$".format(kilometrage, prediction))
	if (theta0 == 0 and theta1 == 0):
		print("Note: it seems that the model is not trained yet. Run train.py to set weights.")

if (__name__ == "__main__"):
	main()