import random
from lilgrad.core import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
    
    
        
class Neuron(Module):
    def __init__(self, n_inputs, activation='linear'):
        self.n_inputs = n_inputs
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

        self.activation = activation

    def __call__(self, x):
        result = sum([self.weights[i]*x[i] for i in range(self.n_inputs)]) + self.bias

        if self.activation == 'relu':
            result = result.relu()
        elif self.activation == 'sigmoid':
            result = result.sigmoid()
        elif self.activation == 'tanh':
            result = result.tanh()
        return result

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self):
        return f'{self.activation} Neuron({self.n_inputs})'

class Layer(Module):
    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f'Layer({self.n_inputs, self.n_outputs})'

class MLP(Module):
    def __init__(self, n_inputs, n_layer_outputs, hidden_activations='relu', output_activation='linear'):
        self.layers = [Layer(n_inputs, n_layer_outputs[0], activation='relu')]

        for i in range(1, len(n_layer_outputs) - 1):
            self.layers.append(Layer(n_layer_outputs[i-1], n_layer_outputs[i], activation=hidden_activations))
        
        self.layers.append(Layer(n_layer_outputs[-2], n_layer_outputs[-1], activation=output_activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def __repr__(self):
        return f'MLP {", ".join(layer.__repr__() for layer in self.layers)}'    