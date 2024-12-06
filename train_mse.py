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

# Normalize features (km) and target (price) using z-score normalization
df['km_normalized'] = (df['km'] - df['km'].mean()) / df['km'].std()
df['price_normalized'] = (df['price'] - df['price'].mean()) / df['price'].std()

print("Normalized DataFrame:\n", df)

# Prepare training data
training_data = []
for feature, target in zip(df['km_normalized'], df['price_normalized']):
    training_data.append((np.array([[feature]]), target))


# Train the network
net.fit(training_data, epochs=50, eta=0.01, validation_data=None, batch_size=1)

# Test feedforward with normalized input
# Assume net_output is the output from the network
# Small test for preprocessing
# Test normalization
test_km = 74000
test_km_normalized = (test_km - df['km'].mean()) / df['km'].std()
net_output = net.feed_forward(np.array([[test_km_normalized]]))

# Denormalize using z-score normalization
denormalized_output = (net_output * df['price'].std()) + df['price'].mean()
print("Denormalized Output:", denormalized_output)

print(f"Weight: {net.layers[-1].weights}")
print(f"Bias: {net.layers[-1].biases}")


# Optionally, save the model
# net.save_model("model.pkl")
