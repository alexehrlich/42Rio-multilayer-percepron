from network import Network, Layer, sigmoid, sigmoid_prime, softmax
import pandas as pd

net = Network() 

net.add_layer(Layer("input", 30, None, None))
net.add_layer(Layer("hidden", 15, sigmoid, sigmoid_prime))
net.add_layer(Layer("output", 2, sigmoid, sigmoid_prime))

for i, 	layer in enumerate(net.layers):
	if layer.weights is None:
		print(f"Input layer {i} has no weigths.")
	else:
		print(f"Shape of weights of layer {i}: {layer.weights.shape}\n")

try:
	features_df_train = pd.read_csv("./csv/created/features_train.csv")
	target_df_train = pd.read_csv("./csv/created/target_train.csv")
	features_df_validation = pd.read_csv("./csv/created/features_val.csv")
	target_df_validation = pd.read_csv("./csv/created/target_val.csv")

	X_train = features_df_train.to_numpy()
	Y_train = target_df_train.to_numpy()
	X_val = features_df_validation.to_numpy()
	Y_val = target_df_validation.to_numpy()

	training_data = []
	for X, Y in zip(X_train, Y_train):
		X_r = X.reshape((30, 1))
		Y_r = Y.reshape((1, 1))
		training_data.append((X_r, Y_r))

	net.fit(training_data, 1000, 10, 0.01)
	net.save_model("model.pkl")

except Exception as e:
	print(e.message)



