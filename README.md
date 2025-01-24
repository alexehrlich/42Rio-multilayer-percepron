
# Multilayer Perceptron

This project demonstrates the implementation of a neural network from scratch **without** any fancy ML libary. It can be trained on any kind of multidimensional data to perform multiclass classification or linear regression. See the basic theory below as well as the two example usages where the neural network was trained
- to classify breast cancer data as benign or malignant. The training is based on a real [data set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) provided by the University of Wisconsin.
- to classify handwritten digits. The training is based on the MNIST dataset

`make setup` to install all dependencies

## Examples
### Cancer classification
- `make canc_preprocess_data`: Cleaning and Normalization of the data. Split the Data into 3 substes of **train** and **validation** for the training process and **test** for testing the prediction
- `make canc_train_classifier`: Train the model with the self written neural_network module
- `make canc_predict_probe`: Run the test data through the net and evaluate its performance

#### Preprocessing
The dataset has a slight imbalance of the B and M classes with ratio of 2:1, which is acceptable. No methods like oversampling or undersampling where applied. The subsets remain roughly the same class distribution. 

```python
---Result of data set split: ---
Train class distribution
        Total: 455
        B(0): 286 (62.9%)
        M(1): 169 (37.1%)

Validation class distribution
        Total: 56
        B(0): 37 (66.1%)
        M(1): 19 (33.9%)

Test class distribution
        Total: 58
        B(0): 34 (58.6%)
        M(1): 24 (41.4%)
```

### Digit classification
- `make mnist_train_classifier`: Loads the the data from the raw file wihtout any libary to train the model with the MNIST data set
- `make mnist_predict_probe`: Draw a digit in pygame. Press 'S' to predict the drawing. Press 'D' to clear the canvas. 

## Theory
### What is a Perceptron?
A perceptron is a fundamental unit of a neural network. It mimics a biological neuron, receiving inputs, applying weights, summing them, and passing the result through an activation function.

### Training
The network is designed modular and the layer structure, the activation function and the cost function can be chosen. The model is trained using mini batch, which are shuffeled in each epoch to prevent pattern learning. 

### General network architecture
This image shows the general data flow from an input vector through the matrices (tensors) of the neural network. The connection from one layer to the next can be represented in a matrix because the input of node is the sum of the products of all nodes with their own weight. The bias of a layer can be represented be a simple column vecotr.

![tensor_flow](./img/layer_flow.jpg) 

With result of the forward pass and the current values of all network parameters (weights and biases) we can calculate how good or bad the result is since we know the label of the passed data from our training dataset. This is done by using a Cost function (See later). We then want to find out how **every** parameter is affecting the cost. Since we want the minimal Cost for the best prediction we need to minimize it. Since the cost depends on every parameter, we need to find the **partial derivative** of the cost function with respect to every parameter in the network. With the partial derivatives we can use gradient descent to make a step towards the optimun as described here.

### Gradient descent
See [here!](https://github.com/alexehrlich/42Rio-AI-ft_linear_regression)

### Backpropagation
With backpropagation we can calculate every derivative of the cost function with respect to all the network parameters. With the derivative we can tweak every parameter with gradient descent. The following shows a simple one dimensional network and how to calculate the partial derivate of the cost function with respect to the parameters of the first layer. That way the error is ***backpropageted** through the net. It turns out that we can mulitply the error by the transposed matrix and then by the derivative of the acitvation function in order to send the error back through the network.
![backprop](./img/backprop.png)

### Cost functions
A **cost function** measures the performance of a machine learning model by quantifying the error between predicted and actual outputs. It serves as the guiding metric to adjust the model's parameters (weights and biases) during training, with the goal of minimizing this error.

Some common cost functions include:

- **Mean Squared Error (MSE)**: Commonly used for regression tasks, it calculates the average squared difference between predicted and actual values.

  ```
  MSE = (1 / n) * Σ (y_i - ŷ_i)²
  ```

- **Cross-Entropy Loss**: Used in classification tasks, especially with probabilistic outputs like softmax, it measures the dissimilarity between predicted probability distributions and actual labels.

  ```
  L = -(1 / n) * Σ [y_i * log(ŷ_i)]
  ```

By minimizing the cost function during training, the model learns to make better predictions. Gradient descent and its variants are often employed to achieve this by iteratively updating model parameters.

### Special case: softmax output layer and Categorical Cross Entropy Loss function
This is a common combination and there is a buty behind it to end up being a simple equation. [This video](https://www.youtube.com/watch?v=znqbtL0fRA0&pp=ygUeY3Jvc3MgZW50cm9weSBsb3NzIGFuZCBzb2Z0bWF4) explains the mathematical proof extremely good.


## Resources

- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com)
- [Gradient Descent, Step by Step (YouTube)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.youtube.com/playlist%3Flist%3DPLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&ved=2ahUKEwjO9O-8vIGLAxVrXfEDHeycIhcQFnoECB0QAQ&usg=AOvVaw1-u2IIZQvYATQYMHEhaCWT)
  


