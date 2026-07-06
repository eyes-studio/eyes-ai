import math
import random

class MathENG1:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def relu(x):
        return max(0, x)
    
    @staticmethod
    def gelu(x):
        return 0.5 * x * (
            1 + math.tanh(
                math.sqrt(2 / math.pi) *
                (x + 0.044715 * x**3)
            )
        )
    
    @staticmethod
    def sigmoid_derivative(x):
        y = 1 / (1 + math.exp(-x))
        return y * (1 - y)
    
    @staticmethod
    def relu_derivative(x):
        if x > 0:
            return 1
        else:
            return 0
    
    @staticmethod
    def gelu_derivative(x):
        a = math.sqrt(2 / math.pi)
    
        tanh_part = math.tanh(
            a * (x + 0.044715 * x**3)
        )

        sech = 1 - tanh_part**2

        return 0.5 * (
            1 + tanh_part +
            x * sech * a * (1 + 3 * 0.044715 * x**2)
        )
    
class Neuron:
    def __init__(self, inputs):
        self.weights = [
            random.uniform(-1, 1)
            for _ in range(inputs)
        ]

        self.bias = random.uniform(-1, 1)
    
    def forward(self, inputs):
        result = self.bias

        for x, w in zip(inputs, self.weights):
            result += x * w

        return result
    
    def backward(self, error, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += (
                error *
                self.last_inputs[i] *
                learning_rate
            )

        self.bias += error * learning_rate

class Layer:
    def __init__(self, input_size, neuron_count):
        self.neurons = []

        for _ in range(neuron_count):
            self.neurons.append(
                Neuron(input_size)
            )


    def forward(self, inputs):
        outputs = []

        for neuron in self.neurons:
            outputs.append(
                neuron.forward(inputs)
            )

        return outputs

    def backward(self, errors, learning_rate):
        for neuron, error in zip(self.neurons, errors):
            neuron.backward(error, learning_rate)


class Network:
    def __init__(self, input_neurons, layers, neurons_per_layer, output_neurons):
        self.input_neurons = input_neurons
        self.layers = []

        for i in range(layers):
            if i == 0:
                input_size = input_neurons
            else:
                input_size = neurons_per_layer

            self.layers.append(
                Layer(input_size, neurons_per_layer)
            )

        self.output_layer = Layer(
            neurons_per_layer,
            output_neurons
        )


    def forward(self, inputs):
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        output = self.output_layer.forward(output)

        return output

    def train(self, inputs, target, epochs=1000, learning_rate=0.01):

        for epoch in range(epochs):

            output = self.forward(inputs)

            errors = []

            for o, t in zip(output, target):
                errors.append(t - o)


            # навчання вихідного шару
            self.output_layer.backward(
                errors,
                learning_rate
            )