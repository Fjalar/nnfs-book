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

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # avoid division with zero

        # handle either categorical labels (just index of correct label),
        # or one-hot encoded labels (zeroes and a one on correct label)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[ # index using correct labels as indices
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
        else:
            raise ValueError

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def main():
    np.random.seed(0) # for reproducibility

    # dataset
    X, y = spiral_data_generator(100, 3)

    dense1 = Layer_Dense(2, 3)

    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(3, 3)

    activation2 = Activation_Softmax()

    loss_function = Loss_CategoricalCrossentropy()

    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    activation2.forward(dense2.output)

    print(activation2.output[:5])

    loss = loss_function.calculate(activation2.output, y)

    print(f"loss: {loss}")

    predictions = np.argmax(activation2.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    print(f"accuracy: {accuracy}")
    
if __name__ == "__main__":
    main()
