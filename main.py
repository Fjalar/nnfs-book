import numpy as np
from nnfs.datasets import spiral_data_generator

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        pass

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

def main():
    np.random.seed(0) # for reproducibility

    # dataset
    X, y = spiral_data_generator(100, 3)

    dense1 = Layer_Dense(2, 3)

    activation1 = Activation_ReLU()

    dense1.forward(X)

    activation1.forward(dense1.output)

    print(activation1.output[:5])

if __name__ == "__main__":
    main()
