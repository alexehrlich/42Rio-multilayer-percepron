from network import Network, Layer
from functions import sigmoid, ReLU, softmax
import pandas as pd

net = Network()
try:
	net.add_layer(Layer("input", 30, activation=None))
	net.add_layer(Layer("hidden", 20, activation=ReLU))
	net.add_layer(Layer("hidden", 16, activation=ReLU))
	net.add_layer(Layer("output", 2, activation=softmax))

except Exception as e:
	print(e.message)
	exit()

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
	validation_data = []
	for X, Y in zip(X_train, Y_train):
		training_data.append((X.reshape((30, 1)), Y[0]))
	for X, Y in zip(X_val, Y_val):
		validation_data.append((X.reshape((30, 1)), Y[0]))

	#net.fit(training_data, epochs=70, eta=0.01, validation_data=validation_data)
	net.fit(training_data, epochs=200, eta=0.0001, validation_data=validation_data)
	net.save_model("model.pkl")

except Exception as e:
	print("Error: ", str(e))
	print("Run <make preprocess_data> first")
	exit()