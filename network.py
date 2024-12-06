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
import pdb

class Model:
	def __init__(self, model_type):
		self.type = model_type
		self.net = None
		self.epochs = None
		self.has_validation_data = False
		self.train_loss_values = []
		self.val_loss_values = []
		self.train_acc_values = []
		self.val_acc_values = []
		self._configure_model()
	
	def _configure_model(self):
		if self.type == 'linear':
			self.net = Network(loss_function=mse_loss)
		elif self.type == 'multi-classifier':
			self.net = Network(loss_function=categorical_cross_entropy_loss)
		else:
			raise ModelTypeError()
	
	def validate(self, data):
		correct_predictions = 0
		validation_loss = 0
		for features, label in data:
			net_out = self.net.feed_forward(features)
			validation_loss += self.net.loss_function(net_out, label)
			predicted = self.net.feed_forward(features)
			result = np.argmax(predicted)
			if result == label:
				correct_predictions += 1
		return ((correct_predictions/len(data)*100, validation_loss/len(data)))
	
	def check_network(self):
		num_layers = len(self.net.layers)

		if num_layers < 1:
			raise NetArchitectureError("The network must have at least 2 Layers. First must be <input>, last must be <output>.")
		
		if self.net.layers[-1].type != "output":
			raise NetArchitectureError("The last layer must be of type <output>.")
		
		if self.net.layers[0].type != "input":
			raise NetArchitectureError("The first layer must be of type <input>.")
		
		if self.type == "linear":
			if num_layers != 2:
				raise NetArchitectureError("Linear regression model must have exactly two layers <input, output>.")
			if self.net.layers[-1].activation is not linear:
				raise ActivationFunctionError("Linear regression must have linear activation on output layer.")
			if self.net.loss_function != mse_loss:
				raise LossFunctionError("Linear regression must have MSE loss function.")

		elif self.type == "multi-classifier":
			if self.net.layers[-1].activation not in [softmax, sigmoid]:
				raise NetArchitectureError("Classification model requires softmax or sigmoid activation on the output layer.")
			if self.net.loss_function != categorical_cross_entropy_loss:
				raise LossFunctionError("Multiclass classification requires categorical cross-entropy loss.")

	def create_mini_batches(self, training_data, batch_size):
		return [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
	
	def fit(self, training_data, epochs, eta, validation_data = None, batch_size=1):
		self.epochs = epochs
		if batch_size < 1:
			raise ValueError("Batch size ust be greater than 0.")
		train_len = len(training_data)
		self.check_network()
		print(f"X_train samples: {train_len}")
		if validation_data:
			print(f"X_val samples: {len(validation_data)}")
		for epoch in range(1, epochs + 1):
			train_loss = 0
			correct_train_prediction = 0
			random.shuffle(training_data)
			for mini_batch in self.create_mini_batches(training_data, batch_size):
				batch_loss, batch_correct_predictions = self.net.learn_mini_batch(mini_batch, eta, batch_size)
				train_loss += batch_loss
				correct_train_prediction += batch_correct_predictions
			if validation_data:
				self.has_validation_data = True
				val_acc, val_loss = self.validate(validation_data)
				self.val_loss_values.append(val_loss)
				self.val_acc_values.append(val_acc)
				print(f"Epoche: {epoch}/{epochs}, Training loss: {train_loss/train_len}, Validation loss: {val_loss}")
			else:
				print("Epoche - ", epoch, ", Training loss: ", train_loss/train_len)
			self.train_loss_values.append(train_loss/train_len)
			self.train_acc_values.append(correct_train_prediction/train_len * 100)

	def plot_training(self):
		if self.type == 'multi-classifier':
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
		elif self.type == 'linear':
			fig, ax1 = plt.subplots(figsize=(14,6))

		x_range = np.arange(1, self.epochs + 1)
		ax1.plot(x_range, self.train_loss_values, label='Training Loss', color='blue')
		if self.has_validation_data:
			ax1.plot(x_range, self.val_loss_values, label='Validation Loss', color='orange')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss')
		ax1.set_title('Loss Over Epochs')
		ax1.legend(loc='best')

		if self.type == "multi-classifier":
			ax2.plot(x_range, self.train_acc_values, label='Training Accuracy', color='green')
			if self.has_validation_data:
				ax2.plot(x_range, self.val_acc_values, label='Validation Accuracy', color='red')
			ax2.set_xlabel('Epochs')
			ax2.set_ylabel('Accuracy (%)')
			ax2.set_title('Accuracy Over Epochs')
			ax2.legend(loc='best')

		plt.tight_layout()
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
		with open(file_name, 'rb') as f:
			return pickle.load(f)


class Network:

	def __init__(self, loss_function):
		self.layers = []
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
			raise NetArchitectureError("First layer must be of type <input>.")
		if self.layers and layer.type == "input":
			raise NetArchitectureError("Net must only have one input layer.")
		if self.layers and self.layers[-1].type == "output":
			raise NetArchitectureError("Net must only have one output layer.")
		self.layers.append(layer)
		if layer.type != "input":
			layer.weight_initialization(layer, self.layers[-2].nodes)
			layer.nabla_w = np.zeros(layer.weights.shape)

	def backpropagation(self, target):
		"""
			Pass the delta backwards. The layer fills its nabla 
			matrix with the gradients of the Cost with
			respect to weights and biases.The gradient is a special
			calculation for the derivative of the Categorical Cross
			Entropy function with respect to the z of the last layer.
			See math here: TODO.
		"""
		if self.loss_function == categorical_cross_entropy_loss:
			delta = self.layers[-1].activations - one_hot(target, self.layers[-1].nodes) #simplified version of CCE loss with softmax
		else:
			delta = self.loss_function_derivative(self.layers[-1].activations, target) * self.layers[-1].derivative_activation(self.layers[-1].z)
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
		if input.shape[0] != self.layers[1].weights.shape[1]:
			raise DimensionError()
		self.layers[0].activations = input
		for i in range(1, len(self.layers)):
			self.layers[i].forward(self.layers[i-1].activations)
		return self.layers[-1].activations

	def learn_mini_batch(self, mini_btach, eta, batch_size):
		batch_loss = 0
		batch_correct_predictions = 0
		for features, target in mini_btach:
			net_out = self.feed_forward(features)
			if self.loss_function == categorical_cross_entropy_loss and np.argmax(net_out) == target:
				batch_correct_predictions += 1
			batch_loss += self.loss_function(net_out, target)
			self.backpropagation(target)
			self.learn_parameter(eta, batch_size)
		return batch_loss, batch_correct_predictions

	

class Layer:
	def __init__(self, layer_type, nodes, activation, weight_initialization):
		"""
			Every derivative dC/dw_ij of the Cost with respect to the weights
			is stored in a nabla matrix with the same shape as the weights matrix.
			Same for the biases vector.
			The inpt is the activation vactor of the prev layer, which is needed
			in the backpropagation.
		"""
		if (layer_type not in ['input', 'hidden', 'output']):
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
