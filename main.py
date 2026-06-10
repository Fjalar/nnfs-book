import numpy as np
from nnfs.datasets import spiral_data_generator

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # normal distribution for weights, zero for biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # subtract largest input to prevent exploding values, doesn't impact result
        # (aside from floating point error)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

def main():
    np.random.seed(0) # for reproducibility

    # dataset
    X, y = spiral_data_generator(100, 3)

    dense1 = Layer_Dense(2, 3)

    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 3)

    activation2 = Activation_Softmax()

    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    activation2.forward(dense2.output)

    print(activation2.output[:5])

if __name__ == "__main__":
    main()
