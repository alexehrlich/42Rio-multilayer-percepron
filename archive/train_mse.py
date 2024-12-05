from network import Network, Layer
from functions import *
import pandas as pd

net = Network(loss_function=mse_loss)

try:
	net.add_layer(Layer("input", 1, activation=None, weight_initialization=None))
	net.add_layer(Layer("hidden", 2, activation=linear, weight_initialization=he_initialization))
	net.add_layer(Layer("hidden", 2, activation=linear, weight_initialization=he_initialization))
	net.add_layer(Layer("output", 1, activation=linear, weight_initialization=he_initialization))
except Exception as e:
	print(e.message)
	exit()

df = pd.read_csv("./csv/house_data.csv")

training_data = []

for feature, target in zip(df['km'], df['price']):
	training_data.append((np.array([[feature]]), target))

print("Net out: ", net.feed_forward(np.array([[2000]])))
net.fit(training_data, epochs=20, eta=0.1, validation_data=None, batch_size=1)
#net.save_model("model.pkl")

