import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import func_deriv, categorial_cross_entropy_loss
import pdb
import random

class Network:
	def __init__(self):
		self.layers = []

	def add_layer(self, layer):
		"""
			Add a layer to the net and connect to the previous,
			by createing a random weight matrix of shape
			(nodes of this layer x nodes of the previous layer).
			Also creates a nabla matrix, to store all the gradients
			of the Cost with respect to the weights.
			Seed used for reproducability.
		"""
		self.layers.append(layer)
		if layer.type != "input":
			np.random.seed(39)
			layer.weights = np.random.randn(layer.nodes, self.layers[-2].nodes)
			layer.nabla_w = np.zeros(layer.weights.shape)

	def backpropagation(self, features, target):
		"""
			Pass the delta backwards. The layer fills its nabla 
			matrix with the gradients of the Cost with
			respect to weights and biases.The gradient is a special
			calculation for the derivative of the Categorial Cross
			Entropy function with respect to the z of the last layer.
			See math here: TODO.
		"""
		net_out = self.feed_forward(features)
		one_hot_target = self.one_hot(target)
		loss = categorial_cross_entropy_loss(net_out, one_hot_target)

		delta = self.layers[-1].activations - one_hot_target
		for layer in reversed(self.layers[1:]):
			delta = layer.backward(delta)
		return loss

	def learn_parameter(self, eta):
		"""
			Update the weights matrix and bias vector with
			Gradient descent. Change the weights/biases in 
			the opposite direction of the slope of that 
			parameter (with the minus)
		"""
		for layer in self.layers[1:]:
			layer.biases -= eta * layer.nabla_b
			layer.weights -= eta * layer.nabla_w

	def feed_forward(self, input):
		"""
			Pass a given input vector through the
			net and return the output of the last layer.
			Since the first input layer has no activation function,
			its activated outpur is the input itself.
		"""
		if len(self.layers) == 0:
			raise EmptyNetworkError()
		if input.shape[0] != self.layers[1].weights.shape[1]:
			raise DimensionError()
		self.layers[0].activations = input
		for i in range(1, len(self.layers)):
			self.layers[i].forward(self.layers[i-1].activations)
		return self.layers[-1].activations

	def one_hot(self, y_train):
		"""
			Encode the target value to make it
			comparable to the networks output vector
		"""
		if y_train == 0:
			return [[1], 
					[0]]
		elif y_train == 1:
			return [[0],
					[1]]

	def validate(self, data):
		right = 0
		for features, label in data:
			predicted = self.feed_forward(features)
			result = np.argmax(predicted)
			if result == label:
				right += 1
		return (f"{(right/len(data)*100):.2f}%")

	def fit(self, training_data, epochs, eta, validation_data = None):
		#TODO: Check layer requirements
		loss_values = []
		for epoch in np.arange(0, epochs):
			loss = 0
			random.seed(42)
			random.shuffle(training_data)
			for features, target in training_data:
				loss += self.backpropagation(features, target)
				self.learn_parameter(eta)
			#update the weights and biases with learn_parameter()
			if validation_data:
				print("Epoche - ", epoch, ", Training CCE-loss: ", loss, ", Valid Accuracy: ", self.validate(validation_data))
			else:
				print("Epoche - ", epoch, ", Training CCE-loss: ", loss)
			loss_values.append(loss)
		# Plot the loss over epochs
		plt.plot(np.arange(0, epochs), loss_values, label='Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Loss Over Epochs')
		plt.legend()
		plt.show()

	def save_model(self, file_name):
		with open(file_name, 'wb') as f:
			pickle.dump(self, f)
	
	@staticmethod
	def load_model(file_name):
		with open(file_name, 'rb') as f:
			return pickle.load(f)

class Layer:
	def __init__(self, layer_type, nodes, activation):
		"""
			Every derivative dC/dw_ij of the Cost with respect to the weights
			is stored in a nabla matrix with the same shape as the weights matrix.
			Same for the biases vector.
			The inpt is the activation vactor of the prev layer, which is needed
			in the backpropagation.
		"""
		self.type = layer_type
		self.nodes = nodes
		self.input = None
		self.activation = activation
		self.derivative_activation = func_deriv[activation]
		self.weights = None
		self.nabla_w = None
		self.biases = np.zeros((nodes, 1))
		self.nabla_b = np.zeros((nodes, 1))
		self.z = np.zeros((nodes, 1))
		self.activations = np.zeros((nodes, 1))

	def forward(self, input):
		"""
			input are the activatons of the previous layer.
		"""
		self.input = input
		self.z = np.dot(self.weights, input) + self.biases
		self.activations = self.activation(self.z)
		return self.activations
	
	# TEMP = multiply the passed gradient with the derivative of the activation with the unactivated output of that layer
	# TEMP is the derivative for the bias --> update the bias
	# multiply TEMP by the input of that node, this is the derivative of the weight --> update the weight
	# multiply TEMP by the OLD weight and return it --> Input for the next layer
	# gradient is a vector with number_rows = number_nodes
	def backward(self, delta):
		#Elementwise multiplication of the passed gradient from previous layer
		# with the derivative of the activation function of that layer.
		if self.type != "output":
			temp = np.multiply(delta, self.derivative_activation(self.z))
		else:
			temp = delta

		self.nabla_b = temp
		self.nabla_w = np.dot(temp, self.input.T)

		#passing as gradient to the previous layer
		return np.dot(self.weights.T, temp)
