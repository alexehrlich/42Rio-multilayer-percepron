import numpy as np

#Elementwise comparison with 0. Returns always the max of the elementwise
#comparison. For negative ones 0, else the the element.
def ReLU(input):
	return np.maximum(input, 0)

def derivative_ReLU(input):
	return np.where(input > 0, 1, 0)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

#Gets a vector and returns a vector
def softmax(input):
	temp = np.exp(input - np.max(input))
	return temp / np.sum(temp)

def categorial_cross_entropy_loss(predictions, targets):
	# Ensure numerical stability by adding epsilon
	epsilon = 1e-15
	predictions = np.clip(predictions, epsilon, 1 - epsilon)

	# Calculate the log of predictions
	log_predictions = np.log(predictions)

	# Element-wise multiplication of log_predictions and targets
	temp = np.multiply(log_predictions, targets)

	# Calculate the cross-entropy loss
	loss = -np.sum(temp)

	return loss

def derivative_crossentropy_softmax(layer_out, target_vec):
	return layer_out - target_vec

def binary_cross_entropy(target_vec, prediction_vec):
	epsilon = 1e-15
	predictions = np.clip(prediction_vec, epsilon, 1 - epsilon)
	N = len(target_vec)
	sum = 0
	for i in range(len(target_vec)):
		sum += target_vec[i][0] * math.log(predictions[i][1]) + (1 - target_vec[i][0]) * math.log(predictions[i][0])
	return -sum/N

func_deriv = {
	None: None,
	ReLU: derivative_ReLU,
	sigmoid: sigmoid_prime,
	softmax: None
}