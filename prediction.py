from nn import Network, Layer
import pandas as pd
import math
import numpy as np

def binary_cross_entropy(target_vec, prediction_vec):
	epsilon = 1e-15
	predictions = np.clip(prediction_vec, epsilon, 1 - epsilon)
	N = len(target_vec)
	sum = 0
	for i in range(len(target_vec)):
		sum += target_vec[i][0] * math.log(predictions[i][1]) + (1 - target_vec[i][0]) * math.log(predictions[i][0])
	return -sum/N

net = Network.load_model("model.pkl")

features_df_test = pd.read_csv("./features_test.csv")
target_df_test = pd.read_csv("./target_test.csv")
X_test = features_df_test.to_numpy()
Y_test = target_df_test.to_numpy()

results = []
for test, target in zip(X_test, Y_test):
	test_c = test.reshape(-1, 1)
	out = net.feed_forward(test_c)

	results.append(out)

print("BCE: ", binary_cross_entropy(Y_test, results))
