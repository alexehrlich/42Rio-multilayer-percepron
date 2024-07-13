from nn import Network, Layer
import pandas as pd

net = Network.load_model("model.pkl")

features_df_test = pd.read_csv("./features_test.csv")
target_df_test = pd.read_csv("./target_test.csv")
X_test = features_df_test.to_numpy()
Y_test = target_df_test.to_numpy()
right = 0
wrong = 0
for test, target in zip(X_test, Y_test):
	test_c = test.reshape(-1, 1)
	out = net.feed_forward(test_c)
	res = 0 if out[0] > out[1] else 1
	print("Target: ", target, "-->", res)
	if target[0] == res:
		right += 1
	else:
		wrong += 1

print("Right: ", right)
print("Wrong: ", wrong)
