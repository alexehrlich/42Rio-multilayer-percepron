import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
from functions import *
import random
from exceptions import *
import os


class Network:
	def __init__(self, loss_function):
		self.layers = []
		self.output_counter = 0
		self.loss_function = loss_function
		self.loss_function_derivative = loss_deriv[loss_function]

	def add_layer(self, layer):
		"""
			Add a layer to the net and connect to the previous,
			by createing a random weight matrix of shape
			(nodes of this layer x nodes of the previous layer).
			Also creates a nabla matrix, to store all the gradients
			of the Cost with respect to the weights.
			Seed used for reproducability.
		"""
		if not self.layers and layer.type != "input":
			raise InputLayerError()
		if self.layers and layer.type == "input":
			raise NetArchitectureError()
		if layer.type == "output":
			self.output_counter += 1
		self.layers.append(layer)
		if layer.type != "input":
			layer.weight_initialization(layer, self.layers[-2].nodes)
			layer.nabla_w = np.zeros(layer.weights.shape)

	def backpropagation(self, one_hot_target):
		"""
			Pass the delta backwards. The layer fills its nabla 
			matrix with the gradients of the Cost with
			respect to weights and biases.The gradient is a special
			calculation for the derivative of the Categorial Cross
			Entropy function with respect to the z of the last layer.
			See math here: TODO.
		"""
		if self.loss_function == categorial_cross_entropy_loss:
			delta = self.layers[-1].activations - one_hot_target #simplified version of CCE loss with softmax
		else:
			delta = self.loss_function_derivative(self.layers[-1].activations, one_hot_target) * self.layers[-1].derivative_activation(self.layers[-1].z)
		for layer in reversed(self.layers[1:]):
			delta = layer.backward(delta)

	def learn_parameter(self, eta, batch_size):
		"""
			Update the weights matrix and bias vector with
			Gradient descent. Change the weights/biases in 
			the opposite direction of the slope of that 
			parameter (with the minus)
		"""
		for layer in self.layers[1:]:
			layer.biases -= eta * (layer.nabla_b/batch_size)
			layer.weights -= eta * (layer.nabla_w/batch_size)
			#reset the nablas after accumulation and learing for the next batch
			layer.nabla_b = np.zeros_like(layer.nabla_b)
			layer.nabla_w = np.zeros_like(layer.nabla_w)

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
		one_hot_vector = np.zeros((self.layers[-1].nodes, 1))
		one_hot_vector[y_train] = 1
		return one_hot_vector

	def validate(self, data):
		right = 0
		validation_loss = 0
		for features, label in data:
			net_out = self.feed_forward(features)
			one_hot_target = self.one_hot(label)
			validation_loss += categorial_cross_entropy_loss(net_out, one_hot_target)
			predicted = self.feed_forward(features)
			result = np.argmax(predicted)
			if result == label:
				right += 1
		return ((right/len(data)*100, validation_loss/len(data)))
	
	def check_network(self):
		num_layers = len(self.layers)
		if num_layers == 0:
			raise EmptyNetworkError()
		if self.layers[-1].type != "output" or self.layers[-1].activation != softmax:
			raise OutputLayerError()
		if num_layers < 4 or self.output_counter > 1:
			raise NetArchitectureError()

	def create_mini_batches(self, training_data, batch_size):
		return [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]

	def learn_mini_batch(self, mini_btach, eta, batch_size):
		batch_loss = 0
		batch_correct_predictions = 0
		for features, target in mini_btach:
			net_out = self.feed_forward(features)
			one_hot_target = self.one_hot(target)
			if np.argmax(net_out) == target:
				batch_correct_predictions += 1
			batch_loss += categorial_cross_entropy_loss(net_out, one_hot_target)
			self.backpropagation(one_hot_target)
			self.learn_parameter(eta, batch_size)
		return batch_loss, batch_correct_predictions

	def fit(self, training_data, epochs, eta, validation_data = None, batch_size=1):
		if batch_size < 1:
			raise ValueError("Batch size ust be greater than 0.")
		train_len = len(training_data)
		self.check_network()
		print(f"X_train samples: {train_len}")
		if validation_data:
			print(f"X_val samples: {len(validation_data)}")
		train_loss_values = []
		val_loss_values = []
		train_acc_values = []
		val_acc_values = []
		for epoch in range(1, epochs + 1):
			train_loss = 0
			correct_train_prediction = 0
			random.shuffle(training_data)
			mini_batches = self.create_mini_batches(training_data, batch_size)
			for mini_batch in mini_batches:
				batch_loss, batch_correct_predictions = self.learn_mini_batch(mini_batch, eta, batch_size)
				train_loss += batch_loss
				correct_train_prediction += batch_correct_predictions
			if validation_data:
				val_acc, val_loss = self.validate(validation_data)
				val_loss_values.append(val_loss)
				val_acc_values.append(val_acc)
				print(f"Epoche: {epoch}/{epochs}, Training CCE loss: {train_loss/train_len}, Validation loss: {val_loss}")
			else:
				print("Epoche - ", epoch, ", Training CCE-loss: ", train_loss/train_len)
			train_loss_values.append(train_loss/train_len)
			train_acc_values.append(correct_train_prediction/train_len * 100)

		# Plot the loss over epochs
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

		# Plotting loss on the first subplot
		ax1.plot(np.arange(1, epochs + 1), train_loss_values, label='Training Loss', color='blue')
		ax1.plot(np.arange(1, epochs + 1), val_loss_values, label='Validation Loss', color='orange')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')
		ax1.set_title('Loss Over Epochs')
		ax1.legend(loc='best')

		# Plotting accuracy on the second subplot
		ax2.plot(np.arange(1, epochs + 1), train_acc_values, label='Training Accuracy', color='green')
		ax2.plot(np.arange(1, epochs + 1), val_acc_values, label='Validation Accuracy', color='red')
		ax2.set_xlabel('Epochs')
		ax2.set_ylabel('Accuracy (%)')
		ax2.set_title('Accuracy Over Epochs')
		ax2.legend(loc='best')

		plt.tight_layout()  # Adjust layout to avoid overlap
		if not os.path.exists("./plots"):
			os.makedirs("./plots")
		plt.savefig('./plots/loss_accuracy_training.png')
		plt.show()


	def save_model(self, file_name):
		with open(file_name, 'wb') as f:
			pickle.dump(self, f)
	
	@staticmethod
	def load_model(file_name):
		if not os.path.exists(file_name):
			raise NoModelError()
			return
		with open(file_name, 'rb') as f:
			return pickle.load(f)

class Layer:
	def __init__(self, layer_type, nodes, activation, weight_initialization):
		"""
			Every derivative dC/dw_ij of the Cost with respect to the weights
			is stored in a nabla matrix with the same shape as the weights matrix.
			Same for the biases vector.
			The inpt is the activation vactor of the prev layer, which is needed
			in the backpropagation.
		"""
		if (layer_type != "input" and layer_type != "hidden" and layer_type != "output"):
			raise LayerTypeError()
		if nodes < 1:
			raise LayerNodeError()
		self.type = layer_type
		self.nodes = nodes
		self.input = None
		self.weight_initialization = weight_initialization
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
		# For the output layer it was already calculated before backpropagation
		if self.type != "output":
			temp = np.multiply(delta, self.derivative_activation(self.z))
		else:
			temp = delta

		self.nabla_b += temp
		self.nabla_w += np.dot(temp, self.input.T)

		#passing as gradient to the previous layer
		return np.dot(self.weights.T, temp)
