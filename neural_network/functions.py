import numpy as np

# Activation Functions and Their Derivatives
def ReLU(z):
	return np.maximum(z, 0)

def derivative_ReLU(z):
	return np.where(z > 0, 1, 0)

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))

def softmax(input):
	temp = np.exp(input - np.max(input))
	return temp / np.sum(temp)

def linear(input):
	return input

def linear_derivation(input):
	return 1

# Loss Functions and Their Derivatives
def categorical_cross_entropy_loss(predictions, target):
	"""
	Sum up the log losses of every output class.
	The epsilon is added for numerical stability.
	"""
	one_hot_target = one_hot(target, len(predictions))
	epsilon = 1e-15
	predictions = np.clip(predictions, epsilon, 1 - epsilon)
	log_predictions = np.log(predictions)
	temp = np.multiply(log_predictions, one_hot_target)
	loss = -np.sum(temp)
	return loss

def derivative_crossentropy_softmax(layer_out, target_vec):
	"""
	Combined version of the categorical loss with the softmax in the output layer.
	Normally calculated as dC/da_last * da_last/dz_last.
	"""
	return layer_out - target_vec

def binary_cross_entropy(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
	return - (y_true * np.log(y_pred[1]) + (1 - y_true) * np.log(y_pred[0]))

def mse_loss(y_pred, y_true):
	"""
	Calculate the Mean Squared Error (MSE) loss.
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	squared_errors = (y_true - y_pred) ** 2
	return 0.5 * np.mean(squared_errors)

def mse_derivative(y_pred, y_true):
	"""
	Compute the derivative of the MSE loss with respect to the predictions.
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	gradient = (y_pred - y_true)
	return gradient

# One-Hot Encoding
def one_hot(label, nodes):
	"""
	Encode the target value to make it comparable to the network's output vector.
	"""
	one_hot_vector = np.zeros((nodes, 1))
	one_hot_vector[label] = 1
	return one_hot_vector

# Initialization Methods for weights matrices
def he_initialization(layer, prev_layer_nodes):
	np.random.seed(39)
	layer.weights = np.random.randn(layer.nodes, prev_layer_nodes) * np.sqrt(2 / prev_layer_nodes)

def xavier_initialization(layer, prev_layer_nodes):
	np.random.seed(39)
	limit = np.sqrt(6 / (layer.nodes + prev_layer_nodes))
	layer.weights = np.random.uniform(-limit, limit, size=(layer.nodes, prev_layer_nodes))

def zero_initialization(layer, prev_layer_nodes):
	layer.weights = np.zeros((layer.nodes, prev_layer_nodes))

INITIALIZERS = [None, he_initialization, xavier_initialization, zero_initialization]

# Mapping activation functions to their derivatives
func_deriv = {
	None: None,
	ReLU: derivative_ReLU,
	sigmoid: sigmoid_derivative,
	linear: linear_derivation,
	softmax: None
}

ACTIVATIONS = [None, ReLU, sigmoid, linear, softmax]

loss_deriv = {
	mse_loss: mse_derivative,
	categorical_cross_entropy_loss: derivative_crossentropy_softmax,
}

LOSS_FUNCTIONS = [mse_loss, categorical_cross_entropy_loss]