class DimensionError(Exception):
	def __init__(self):
		self.message = "Wrong Dimension of input vector"

class EmptyNetworkError(Exception):
	def __init__(self):
		self.message = "Network has no layers."

class LayerTypeError(Exception):
	def __init__(self):
		self.message = "Layer must be of type <input>, <hidden> or <output>"

class InputLayerError(Exception):
	def __init__(self):
		self.message = "First layer must be of type <input>"

class OutputLayerError(Exception):
	def __init__(self):
		self.message = "Last layer must be of type <output> with softmax activation"

class LayerNodeError(Exception):
	def __init__(self):
		self.message = "Layer must have at least one node"

class NetArchitectureError(Exception):
	def __init__(self):
		self.message = "Network must have at least two hidden layers and only one input and only one output layer"

class NoModelError(Exception):
	def __init__(self):
		self.message = "There is no model present. Run <make train> first"