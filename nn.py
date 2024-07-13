import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Elementwise comparison with 0. Returns always the max of the elementwise
#comparison. For negative ones 0, else the the element.
def ReLU(input):
	return np.maximum(input, 0)

def derivative_ReLU(input):
	return np.where(input > 0, 1, 0)

#Gets a vector and returns a vector
def softmax(input):
	#print(input)
	temp = np.exp(input - np.max(input))
	#print(temp)
	return temp / np.sum(temp)

def cross_entropy_loss(predictions, targets):
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



#The input vector has a num_input(row) x 1(col) dimension
#The weight matrix has a num_nodes(row) x num_input(col) dimension
#The bias vector has a num_nodes(row) x 1(col) dimension
class Layer:
	layer_index = 1
	def __init__(self, num_nodes, num_input, activation, d_activation):
		self.activation = activation
		self.derivative_activation = d_activation
		self.input = []
		self.weights = np.random.randn(num_nodes, num_input)
		self.biases = np.zeros((num_nodes, 1))
		self.unactivated_out = np.zeros((num_nodes, 1))
		self.activated_out = np.zeros((num_nodes, 1))

		self.layer_index = Layer.layer_index
		Layer.layer_index += 1

		#print("Weight matrix of layer ", self.layer_index)
		#print(self.weights)
		#print("\n")

	def forward(self, input):
		self.input = input
		self.unactivated_out = np.dot(self.weights, input) + self.biases
		self.activated_out = self.activation(self.unactivated_out)
		return self.activated_out
	
	# TEMP = multiply the passed gradient with the derivative of the activation with the unactivated output of that layer
	# TEMP is the derivative for the bias --> update the bias
	# multiply TEMP by the input of that node, this is the derivative of the weight --> update the weight
	# multiply TEMP by the OLD weight and return it --> Input for the next layer
	# gradient is a vector with number_rows = number_nodes
	def backward(self, gradient, learning_rate, flag=1):
		#backup the original weights for passing to the next layer
		old_weights = self.weights

		#Elementwise multiplication of the passed gradient from previous layer
		# with the derivative of the activation function of that layer.
		if flag:
			temp = np.multiply(gradient, self.derivative_activation(self.unactivated_out))
		else:
			temp = gradient

		#Learning process of weights and biases
		#print("Temp: ", temp)
		self.biases = self.biases - learning_rate * temp
		self.weights = self.weights - learning_rate * np.dot(temp, self.input.T)

		#passing as gradient to the previous layer
		return np.dot(old_weights.T, temp)

class DimensionError(Exception):
	def __init__(self):
		self.message = "Wrong Dimension of input vector"
	
class EmptyNetworkError(Exception):
	def __init__(self):
		self.message = "Network has no layers."

class Network:
	def __init__(self):
		self.layers = []
	
	def add_layer(self, layer):
		self.layers.append(layer)
	
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

	def fit(self, x_train, y_train, epochs=1000, learning_rate=0.01):
		loss_values = []
		#iterate over the epochs
		for epoch in np.arange(0, epochs):
			loss = 0
			#iterate over all train samples in one epoch
			for (sample, target) in zip(x_train, y_train):
				sample_column = sample.reshape(-1, 1)
				#make the forwardfeed to get the output for the current weights and biases
				out = self.feed_forward(sample_column)
				#backpropagation
				one_hot_target = self.one_hot(target)
				loss = loss + cross_entropy_loss(out, one_hot_target)

				#Derivation of Cross Entropy comined with Softmax (dL/dInput_k = Output_k - y_k) for that node
				# Input_k (unactivated weighted sum of that node), Output_k (activated Input), 
				# y_k (Entry in the on hot encoded target vector for that node, 0 or 1)
				gradient = self.layers[-1].activated_out - one_hot_target
				flag = 0
				for layer in reversed(self.layers):
					#print("Gradient:\n", gradient)
					gradient = layer.backward(gradient, learning_rate, flag)
					flag = 1
			
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
