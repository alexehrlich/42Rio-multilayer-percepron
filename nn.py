import numpy as np

#Elementwise comparison with 0. Returns always the max of the elementwise
#comparison. For negative ones 0, else the the element.
def ReLU(input):
    return np.maximum(input, 0)

#The input vector has a num_input(row) x 1(col) dimension
#The weight matrix has a num_nodes(row) x num_input(col) dimension
#The bias vector has a num_nodes(row) x 1(col) dimension
class Layer:

    layer_index = 1
    def __init__(self, num_nodes, num_input, activation):
        self.activation = activation
        self.weights = np.ones((num_nodes, num_input))
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
        return self.unactivated_out

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

net = Network()

input = np.array([[3, 4]]).T
print("Input Column Vector: ")
print(input)
print("\n")

net.add_layer(Layer(3, 2, ReLU))
net.add_layer(Layer(1, 3, ReLU))

try:
    print(net.feed_forward(input))
except Exception as e:
    print(e.message)


