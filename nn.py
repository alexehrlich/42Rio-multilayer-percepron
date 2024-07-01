import numpy as np

#Elementwise comparison with 0. Returns always the max of the elementwise
#comparison. For negative ones 0, else the the element.
def ReLU(input):
	return np.maximum(input, 0)

def derivative_ReLU(input):
	return np.where(input > 0, 1, 0)

def softmax(input):
	temp = np.exp(input)
	return temp / np.sum(temp)

def derivative_softmax(input):
	#TBD
	return 1

def loss(target, prediction):
	return 0.5 * ((target - prediction) ** 2)

# The derivative of the loss function with respect to the predicted value
def derivative_loss(target, prediction):
	#-(target - prediction) or
	return prediction - target



#The input vector has a num_input(row) x 1(col) dimension
#The weight matrix has a num_nodes(row) x num_input(col) dimension
#The bias vector has a num_nodes(row) x 1(col) dimension
class Layer:
	layer_index = 1
	def __init__(self, num_nodes, num_input, activation):
		self.activation = activation
		self.weights = np.randomn.randn(num_nodes, num_input)
		self.biases = np.zeros((num_nodes, 1))
		self.unactivated_out = np.zeros((num_nodes, 1))
		self.activated_out = np.zeros((num_nodes, 1))

		self.layer_index = Layer.layer_index
		Layer.layer_index += 1

		print("Weight matrix of layer ", self.layer_index)
		print(self.weights)
		print("\n")

	def feed(self, input):
		self.unactivated_out = np.dot(self.weights, input) + self.biases
		self.activated_out = self.activation(self.unactivated_out)
		return self.activated_out

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
			print("Current input: \n", temp_in_vector)
			temp_in_vector = layer.feed(temp_in_vector)
			print("transformed input: \n", temp_in_vector)
		return temp_in_vector
	
	def fit(self, x_train, y_train, epochs=1000, learning_rate=0.01):
		
		#iterate over the epochs
		for epoch in np.arange(0, epochs):

			#iterate over all train samples in one epoch
			for (sample, target) in zip(x_train, y_train):

				#make the forwardfeed to get the output for the current weights and biases
				out = self.feed_forward(sample.T)

				#calculate the error/loss for the current network configuration
				error = target - out

				#How do I apply the chainrule to adjust the weights and Biases in this generally structured network?

	


net = Network()

#Transponse the input to be a column vector for the multiplication
# with the weight matrix
input = np.array([[3, 4]]).T
print("Input Column Vector: ")
print(input)
print("\n")

#create hidden layers with ReLU activation
net.add_layer(Layer(3, 2, ReLU))
net.add_layer(Layer(3, 3, ReLU))

#add an output layer with the softmax probability distribution
net.add_layer(Layer(2, 3, softmax))

try:
	out = net.feed_forward(input)
	print("Output: \n", out)
except Exception as e:
	print(e.message)


