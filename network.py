import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from functions import func_deriv, cross_entropy_loss
import pdb
import random

class Network:
	def __init__(self):
		self.layers = []

	def add_layer(self, layer):
		#add layer and connect the layer to the previous one
		# by creating a random init weight matrix(seed)
		#Also create the weight matrix, the gradiet weight matrix with same shape and
		self.layers.append(layer)
		if layer.type != "input":
			np.random.seed(39)
			layer.weights = np.random.randn(layer.nodes, self.layers[-2].nodes)
			layer.nabla_w = np.zeros(layer.weights.shape)

	#returns the gradient of the cost with respect to all weigths
	# and biases. The gradients are normalized to the batch size.
	def backpropagation(self, features, target):
		#loop through the layers and set the the gradients to the layers nabla matrix
		net_out = self.feed_forward(features)
		one_hot_target = self.one_hot(target)
		loss = cross_entropy_loss(net_out, one_hot_target)
		#calculate the gradients (weigths and biases) for each layer with backpropagation

		#The first is special, becuase of combination of CCE and softmax
		gradient = self.layers[-1].activations - one_hot_target
		flag = 0
		for layer in reversed(self.layers[1:]):
			gradient = layer.backward(gradient, flag)
			flag = 1
		return loss

	def learn_parameter(self, eta):
		for layer in self.layers[1:]:
			layer.biases -= eta * layer.nabla_b
			layer.weights -= eta * layer.nabla_w

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

	#training data is a tuple of(featrues, target)
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
			print("Epoche - ", epoch, ", loss: ", loss)
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
		self.type = layer_type
		self.nodes = nodes
		self.input = None
		self.activation = activation
		self.derivative_activation = func_deriv[activation]
		self.weights = None
		self.nabla_w = None
		self.biases = np.zeros((nodes, 1)) #col vector of the biases
		self.nabla_b = np.zeros((nodes, 1))
		self.z = np.zeros((nodes, 1)) #col vector of z's (weighted sums)
		self.activations = np.zeros((nodes, 1)) #col vector of the activations

	def forward(self, input):
		self.input = input
		self.z = np.dot(self.weights, input) + self.biases
		self.activations = self.activation(self.z)
		return self.activations
	
	# TEMP = multiply the passed gradient with the derivative of the activation with the unactivated output of that layer
	# TEMP is the derivative for the bias --> update the bias
	# multiply TEMP by the input of that node, this is the derivative of the weight --> update the weight
	# multiply TEMP by the OLD weight and return it --> Input for the next layer
	# gradient is a vector with number_rows = number_nodes
	def backward(self, gradient, flag=1):
		#Elementwise multiplication of the passed gradient from previous layer
		# with the derivative of the activation function of that layer.
		if flag:
			temp = np.multiply(gradient, self.derivative_activation(self.z))
		else:
			temp = gradient

		#Learning process of weights and biases
		#print("Temp: ", temp)
		self.nabla_b = temp
		self.nabla_w = np.dot(temp, self.input.T)
		#self.biases = self.biases - learning_rate * temp
		#self.weights = self.weights - learning_rate * np.dot(temp, self.input.T)

		#passing as gradient to the previous layer
		return np.dot(self.weights.T, temp)
