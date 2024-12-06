import numpy as np

#Elementwise comparison with 0. Returns always the max of the elementwise
#comparison. For negative ones 0, else the the element.
def ReLU(z):
	return np.maximum(z, 0)

def derivative_ReLU(z):
	return np.where(z > 0, 1, 0)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
	return sigmoid(z)*(1-sigmoid(z))

def softmax(input):
	temp = np.exp(input - np.max(input))
	return temp / np.sum(temp)

def linear(input):
	return input

def linear_derivation(input):
	return 1

def one_hot(label, nodes):
	"""
		Encode the target value to make it
		comparable to the networks output vector
	"""
	one_hot_vector = np.zeros((nodes, 1))
	one_hot_vector[label] = 1
	return one_hot_vector

def categorial_cross_entropy_loss(predictions, target):
	# Ensure numerical stability by adding epsilon
	one_hot_target = one_hot(target, len(predictions))
	epsilon = 1e-15
	predictions = np.clip(predictions, epsilon, 1 - epsilon)

	# Calculate the log of predictions
	log_predictions = np.log(predictions)

	# Element-wise multiplication of log_predictions and targets
	temp = np.multiply(log_predictions, one_hot_target)

	# Calculate the cross-entropy loss
	loss = -np.sum(temp)

	return loss

def derivative_crossentropy_softmax(layer_out, target_vec):
	return layer_out - target_vec

def binary_cross_entropy(y_true, y_pred):
	epsilon = 1e-15
	y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
	return - (y_true * np.log(y_pred[1]) + (1 - y_true) * np.log(y_pred[0]))

def mse_loss(y_pred, y_true):
	"""
	Calculate the Mean Squared Error (MSE) loss.
	
	Parameters:
		y_true (list or numpy array): The true target values.
		y_pred (list or numpy array): The predicted values.
		
	Returns:
		float: The MSE loss.
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	squared_errors = (y_true - y_pred) ** 2
	return 0.5 * np.mean(squared_errors)

def mse_derivative(y_pred, y_true):
	"""
	Compute the derivative of the MSE loss with respect to the predictions.
	
	Parameters:
		y_true (numpy array): The true target values.
		y_pred (numpy array): The predicted values.
		
	Returns:
		numpy array: The gradient of the MSE loss.
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	
	# Compute the gradient
	gradient = (y_pred - y_true)
	return gradient



func_deriv = {
	None: None,
	ReLU: derivative_ReLU,
	sigmoid: sigmoid_derivative,
	linear: linear_derivation,
	softmax: None
}

loss_deriv = {
	mse_loss: mse_derivative,
	categorial_cross_entropy_loss: derivative_crossentropy_softmax,
}

def he_initialization(layer, prev_layer_nodes):
	np.random.seed(39)
	layer.weights = np.random.randn(layer.nodes, prev_layer_nodes) * np.sqrt(2 / prev_layer_nodes)

def xavier_initialization(layer, prev_layer_nodes):
	np.random.seed(39)
	limit = np.sqrt(6 / (layer.nodes + prev_layer_nodes))
	layer.weights = np.random.uniform(-limit, limit, size=(layer.nodes, prev_layer_nodes))

def zero_initialization(layer, prev_layer_nodes):
	layer.weights = np.zeros((layer.nodes, prev_layer_nodes))