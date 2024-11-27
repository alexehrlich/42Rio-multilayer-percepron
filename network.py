import random
import numpy as np
import pickle
import pdb

def softmax(input):
	temp = np.exp(input - np.max(input))
	return temp / np.sum(temp)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

class DimensionError(Exception):
	def __init__(self):
		self.message = "Wrong Dimension of input vector"

class EmptyNetworkError(Exception):
	def __init__(self):
		self.message = "Network has no layers."


class Network:
	def __init__(self):
		self.layers = []
	
	def connect_layers(self):
		for i, layer in enumerate(self.layers):
			if layer.type != "input" and layer.weights is None:
				layer.weights = np.random.randn(layer.nodes, self.layers[i - 1].nodes)
				layer.nabla_w = np.zeros(layer.weights.shape)
	
	def add_layer(self, layer):
		self.layers.append(layer)
		self.connect_layers()
	
	def feed_forward(self, input):
		if len(self.layers) == 0:
			raise EmptyNetworkError()
		if input.shape[0] != self.layers[1].weights.shape[1]:
			raise DimensionError()
		#The output of the input layer is the input itself
		self.layers[0].activations = input
		for i in range(1, len(self.layers)):
			self.layers[i].forward(self.layers[i-1].activations)
		return self.layers[-1].activations

	def one_hot(self, y_train):
		if y_train == 0:
			return [[1], 
					[0]]
		elif y_train == 1:
			return [[0],
					[1]]

	def fit(self, training_data, epochs, mini_batch_size, eta, validation_data=None):
		if validation_data:
			n_validation_data = len(validation_data)
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if validation_data:
				print("Epoch {0}: {1} / {2}".format(j, self.evaluate(validation_data), n_validation_data))
			else:
				print("Epoch {0} complete".format(j))

	def backpropagation(self, x, y):
		temp_nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]
		temp_nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]

		net_out = self.feed_forward(x)

		#calcualte the delta for the last layer derivative_crossentropy_softmax:
		delta = net_out - y

		temp_nabla_b[-1] = delta
		temp_nabla_w[-1] = np.dot(delta, self.layers[-2].activations.transpose())

		for l in range(2, len(self.layers)):
			z = self.layers[-l].z
			sp = sigmoid_prime(z) #use the passed derivative later!!
			delta = np.dot(self.layers[-l + 1].weights.transpose(), delta) * sp
			temp_nabla_b[-l] = delta
			temp_nabla_w[-l] = np.dot(delta, self.layers[-l-1].activations.transpose())
		return (temp_nabla_b, temp_nabla_w)

	def update_mini_batch(self, mini_batch, eta):
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
			for layer, dnb, dnw in zip(self.layers[1:], delta_nabla_b, delta_nabla_w):
				layer.nabla_b += dnb
				layer.nabla_w += dnw
		for layer in self.layers[1:]:
			layer.weights = layer.weights - (eta/len(mini_batch)*layer.nabla_w)
			layer.biases = layer.biases - (eta/len(mini_batch)*layer.nabla_b)
	
	def save_model(self, file_name):
		with open(file_name, 'wb') as f:
			pickle.dump(self, f)
	
	@staticmethod
	def load_model(file_name):
		with open(file_name, 'rb') as f:
			return pickle.load(f)

class Layer:
	def __init__(self, type, nodes, activation, derivative_activation):
		self.type = type
		self.nodes = nodes
		self.activation = activation
		self.derivative_activation = derivative_activation
		self.weights = None
		self.nabla_w = None
		self.biases = np.zeros((nodes, 1)) #col vector of the biases
		self.nabla_b = np.zeros((nodes, 1))
		self.z = np.zeros((nodes, 1)) #col vector of z's (weighted sums)
		self.activations = np.zeros((nodes, 1)) #col vector of the activations

	def forward(self, input):
		self.z = np.dot(self.weights, input) + self.biases
		self.activations = self.activation(self.z)
		return self.activations
