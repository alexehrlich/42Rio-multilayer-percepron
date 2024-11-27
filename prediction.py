from network import Network, Layer
import pandas as pd
import math
import numpy as np
from functions import binary_cross_entropy

net = Network.load_model("model.pkl")

features_df_test = pd.read_csv("./csv/created/features_test.csv")
target_df_test = pd.read_csv("./csv/created/target_test.csv")
X_test = features_df_test.to_numpy()
Y_test = target_df_test.to_numpy()

results = []
total = len(X_test)
right = 0
wrong = 0
for test, target in zip(X_test, Y_test):
	test_c = test.reshape(-1, 1)
	out = net.feed_forward(test_c)
	results.append(out)
	if out[0] > out [1]:
		result = 0
	else:
		result = 1
	if target[0] == result:
		right += 1
	else:
		wrong += 1
print(f"Total: {total}")
print(f"Wrong: {wrong}, {(wrong/total*100):.2f}%")
print(f"Right: {right}, {(right/total*100):.2f}%")


