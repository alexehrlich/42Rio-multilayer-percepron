
# Multilayer Perceptron

This project demonstrates the implementation of a neural network from scratch to classify breast cancer data as benign or malignant. The model employs backpropagation and gradient descent for training, based on a real [data set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) provided by the University of Wisconsin.

---

## Usage

1. **Setup**  
   Run the following command to create a virtual environment and install all the required dependencies:
   ```bash
   make setup
   ```

2. **Preprocess Data**  
   Split the dataset into training, validation, and test subsets with a specified ratio:
   ```bash
   make preprocess_data
   ```

3. **Train the Classifier**  
   Train the multilayer neural network for classifying breast cancer data:
   ```bash
   make train_classifier
   ```

4. **Train for Linear Regression**  
   Train a simple neural network for a single-input single-output linear regression model:
   ```bash
   make train_car_linreg
   ```

5. **Make Predictions**  
   Use the test dataset to evaluate the trained classifier on unseen data:
   ```bash
   make prediction
   ```

---

## Neural Networks

### What is a Perceptron?  
A perceptron is a fundamental unit of a neural network. It mimics a biological neuron, receiving inputs, applying weights, summing them, and passing the result through an activation function.

### Neural Network Structure  
A neural network consists of layers of perceptrons (neurons), each with weights and biases. The network propagates data forward from input to output, learning patterns from data.  

*(Placeholder for neural network structure diagram)*

### Matrix and Vector Representation  
In a neural network, weights and biases are represented as matrices and vectors. During a forward pass, input data is multiplied with weights, added to biases, and transformed through activation functions.  

*(Placeholder for forward pass diagram)*

### Loss Functions  
Loss functions measure how well the neural network predicts the desired output. Examples include:  
- Mean Squared Error (MSE) for regression tasks  
- Categorical Cross-Entropy for classification tasks  

---

## Backpropagation

### Why?  
Backpropagation allows the network to learn by updating weights and biases based on the error between predicted and actual outputs. It calculates the gradient of the loss function with respect to each weight through reverse traversal of the network.

### What?  
Backpropagation combines:  
1. The **chain rule of calculus** to compute derivatives.  
2. Gradient descent to optimize weights iteratively.  

---

## Proof of Softmax in Combination with Categorical Cross-Entropy Loss  

The softmax function transforms raw scores (logits) into probabilities:
\[
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

When combined with the categorical cross-entropy loss:
\[
\text{Loss} = -\sum_{i} y_i \log(\sigma(z_i))
\]

Substituting the softmax equation:
\[
\text{Loss} = -\sum_{i} y_i \log\left(\frac{e^{z_i}}{\sum_{j} e^{z_j}}\right)
\]

Simplifies to:
\[
\text{Loss} = -\sum_{i} y_i \left(z_i - \log\sum_{j} e^{z_j}\right)
\]

Further derivations show that the gradients for this loss are simpler and computationally efficient.

*(Detailed calculations can be added here)*

---

## Resources

- [Gradient Descent, Step by Step (YouTube)](https://www.youtube.com/watch?v=sDv4f4s2SB8)

--- 
