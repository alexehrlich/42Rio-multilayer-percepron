import numpy as np
import random

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
			if layer.type != "input" and layer.weights == None:
				layer.weights = np.random.randn(layer.nodes, self.layers[i - 1].nodes)
				layer.nabla_w = np.zeros(layer.weights.shape)
	
	def add_layer(self, layer):
		self.layers.append(layer)
		self.connect_layers()
	
	def feed_forward(self, input):
		if len(self.layers) == 0:
			raise EmptyNetworkError()
		if input.shape[0] != self.layers[0].weights.shape[1]:
			raise DimensionError()
		temp_in_vector = input
		for layer in self.layers:
			temp_in_vector = layer.forward(temp_in_vector)
		return temp_in_vector

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
		batch_nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
		batch_nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

		self.feed_forward(x)

	def update_mini_batch(self, mini_batch, eta):
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
			for layer, dnb, dnw in zip(self.layers, delta_nabla_b, delta_nabla_w):
				layer.nabla_b += dnb
				layer.nabla_w += dnw
		for layer in self.layers:
			layer.weights = layer.weights - (eta/len(mini_batch)*layer.nabla_w)
			layer.biases = layer.biases - (eta/len(mini_batch)*layer.nabla_b)


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
