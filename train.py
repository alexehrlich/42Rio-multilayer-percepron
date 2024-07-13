from nn import Network, Layer, ReLU, derivative_crossentropy_softmax, softmax, derivative_ReLU
import pandas as pd

net = Network()

#create hidden layers with ReLU activation
net.add_layer(Layer(64, 30, ReLU, derivative_ReLU))
net.add_layer(Layer(32, 64, ReLU, derivative_ReLU))

#add an output layer with the softmax probability distribution
net.add_layer(Layer(2, 32, softmax, derivative_crossentropy_softmax))

try:
	features_df_train = pd.read_csv("./features_train.csv")
	target_df_train = pd.read_csv("./target_train.csv")

	X_train = features_df_train.to_numpy()
	Y_train = target_df_train.to_numpy()

	net.fit(X_train, Y_train)
	net.save_model("model.pkl")

except Exception as e:
	print(e.message)