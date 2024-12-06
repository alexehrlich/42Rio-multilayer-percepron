class DimensionError(Exception):
	def __init__(self):
		self.message = "Input vector dimension does not match first layer."

class LayerTypeError(Exception):
	def __init__(self):
		self.message = "Layer must be of type <input>, <hidden> or <output>"

class ActivationFunctionError(Exception):
	def __init__(self, msg):
		self.message = msg

class LayerNodeError(Exception):
	def __init__(self):
		self.message = "Layer must have at least one node"

class NetArchitectureError(Exception):
	def __init__(self, msg):
		self.message = msg

class LossFunctionErrorError(Exception):
	def __init__(self, msg):
		self.message = msg

class NoModelError(Exception):
	def __init__(self):
		self.message = "There is no model present. Run <make train> first"

class ModelTypeError(Exception):
	def __init__(self):
		self.message = "Model type is not supported."