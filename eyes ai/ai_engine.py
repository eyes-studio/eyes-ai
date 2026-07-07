import math
import random
#math core
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
    
    @staticmethod
    def mse_loss(output, target):
        return sum((o - t)**2 for o, t in zip(output, target)) / len(output)
    
#ai core
class Neuron:
    def __init__(self, inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(inputs)]
        self.bias = random.uniform(-1, 1)
        self.last_inputs = None
        self.last_output = None 
    
    def forward(self, inputs, activation=None):
        self.last_inputs = inputs 
        
        result = self.bias
        for x, w in zip(inputs, self.weights):
            result += x * w
        
        self.last_output = result
        
        if activation:
            result = activation(result) 
        
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


    def forward(self, inputs, activation=None):
        self.last_inputs = inputs
        self.activation = activation
        outputs = []
        
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs, activation))
        
        return outputs

    def backward(self, errors, learning_rate, activation_derivative=None):

        if activation_derivative:
            errors = [
                e * activation_derivative(neuron.last_output)
                for e, neuron in zip(errors, self.neurons)
            ]
        
        for neuron, error in zip(self.neurons, errors):
            neuron.backward(error, learning_rate)
        
        return self._compute_input_errors(errors)

    def _compute_input_errors(self, errors):
        input_errors = [0] * len(self.last_inputs)
        
        for neuron, error in zip(self.neurons, errors):
            for i, weight in enumerate(neuron.weights):
                input_errors[i] += error * weight
        
        return input_errors


class Network:
    def __init__(self, layer_sizes, activation='gelu'):
        self.layers = []
        self.activation_name = activation
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(layer_sizes[i], layer_sizes[i+1])
            )


    def forward(self, inputs):
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        output = self.output_layer.forward(output)

        return output

    def train(self, X_train, Y_train, epochs=1000, learning_rate=0.01):
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for inputs, target in zip(X_train, Y_train):
                output = self.forward(inputs)
                loss = MathENG1.mse_loss(output, target)
                total_loss += loss
                
                self._backward(target, learning_rate)
            
            losses.append(total_loss / len(X_train))
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}")
        
        return losses

    def _backward(self, target, learning_rate):
        errors = [t - o for t, o in zip(target, self.forward_output)]
        
        for layer in reversed(self.layers):
            errors = layer.backward(errors, learning_rate)
