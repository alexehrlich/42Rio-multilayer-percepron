from network import Network, Layer
from functions import sigmoid, ReLU, softmax, xavier_initialization, he_initialization
import pandas as pd

net = Network()
try:
	net.add_layer(Layer("input", 30, activation=None, weight_initialization=None))
	net.add_layer(Layer("hidden", 20, activation=ReLU, weight_initialization=he_initialization))
	net.add_layer(Layer("hidden", 16, activation=ReLU, weight_initialization=he_initialization))
	net.add_layer(Layer("output", 2, activation=softmax, weight_initialization=he_initialization))

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

	#Training data needs to be a list of tupels where each tuple is (column_vec of features, int of label).
	# Example. Image of a nine: ([210, 255, 123, ...], 9)
	# The label is used for hot encodig internally.
	net.fit(training_data, epochs=100, eta=0.0001, validation_data=validation_data, batch_size=1)
	net.save_model("model.pkl")

except ValueError as ve:
	print("Value Error:", str(ve))

except FileNotFoundError as fnf:
	print("Run <make preprocess_data> first.")

except Exception as e:
	print("An unexpected error occurred:", str(e))

finally:
	exit()