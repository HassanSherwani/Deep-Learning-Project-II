"""
Basic Neural Network
"""
import warnings
warnings.filterwarnings('ignore')

from numpy import *

class NeuralNetwork():
    # PRIVATE
    def __init__(self):
        # Seed the random number generator, so it generates the
        # same numbers every time the program runs
        random.seed(1)

        # Model a single neuron (3 input -> 1 output):
        # input ]
        # input ] -- > output
        # input ]

        # Random weight for the 3 x 1 matrix with values between -1 to 1 and mean 0
        self.synaptic_weights = 20 * random.random((3,1)) - 10

    # PUBLIC
    # S shaped curve, we pass the weighted sum of the inputs to normalize between 0 and 1
    @classmethod
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivate/gradient of the Sigmoid function
    # Indicates how confident we are about the weights
    @classmethod
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network through trial and error
    # Adjusting the weight in each iteration
    @classmethod
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for _ in range(number_of_training_iterations):
            # Pass the training input through our neural network
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the real output and the predicted one)
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more (because we multiply for the gradient)
            # This means inputs, which are 0, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights
            self.synaptic_weights += adjustment

    def think(self, inputs):
        # Pass inputs through our neural network
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
    # Initialize a single neural network
    neural_network = NeuralNetwork()

    # Print the synaptic weight of the neuron (initialized as random)
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # Our training set has 4 input arrays of 3 values each
    # and one output value for each input
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T      # What does this T mean? Transposed.

    # Train the neuran network using a training set.
    # Do it 10.000 times and make small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    # Print the synaptic weight of the neuron (after training)
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation -> ?: ")
    newtest = array(split(int(input())))
    print(neural_network.think(newtest))