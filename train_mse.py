import numpy as np
import pandas as pd
from network import Network, Layer
from functions import *

# Initialize Network
net = Network(loss_function=mse_loss)

try:
    net.add_layer(Layer("input", 1, activation=None, weight_initialization=None))
    net.add_layer(Layer("output", 1, activation=linear, weight_initialization=he_initialization))
except Exception as e:
    print(e)
    exit()

# Load data
df = pd.read_csv("./csv/car_data.csv")

# Normalize features (km) and target (price)
df['km_normalized'] = (df['km'] - df['km'].min()) / (df['km'].max() - df['km'].min())
df['price_normalized'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())

print(df)
# Prepare training data
training_data = []
for feature, target in zip(df['km_normalized'], df['price_normalized']):
    training_data.append((np.array([[feature]]), target))

# Feed forward a normalized value for testing
normalized_input = (2000 - df['km'].min()) / (df['km'].max() - df['km'].min())
print("Net out: ", net.feed_forward(np.array([[normalized_input]])))

# Train the network
net.fit(training_data, epochs=20, eta=0.00001, validation_data=None, batch_size=1)

# Optionally, save the model
# net.save_model("model.pkl")
