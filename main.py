import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self,input,output):
        print("Creamos la capa de entrada")
        self.input_layer = Layer(input)
        print("Creamos la capa de salida")
        self.output_layer = Layer(output, self.input_layer)
        self.hidden_layers = []


    def add_hidden_layer(self, num_neurons):
        if not self.hidden_layers:
            print("Recoge la capa de entrada")
            back_layer = Layer(num_neurons, self.input_layer)
        else:
            print("Recoge la ultima capa oculta")
            back_layer = Layer(num_neurons, self.hidden_layers[-1])

        print("Rehacemos la capa de salida")
        self.output_layer = Layer(len(self.output_layer.neurons),back_layer)
        self.hidden_layers.append(back_layer)

    def draw_network(self, left, right, bottom, top):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')

        layer_sizes = np.array([len(self.input_layer.neurons)])
        
        for i in self.hidden_layers:
            layer_sizes = np.append(layer_sizes, len(i.neurons))
        
        layer_sizes = np.append(layer_sizes, len(self.output_layer.neurons))

        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)

        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                    [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                    ax.add_artist(line)

        plt.show()
        # fig.savefig('nn.png')


class Layer(object):

    def __init__(self, num_neurons, prev_layer = None):
        self.prev_layer = prev_layer
        if prev_layer:
            self.neurons = [Neuron(len(self.prev_layer.neurons)) for i in range(num_neurons)]
        else:
            self.neurons = [Neuron() for i in range(num_neurons)]


class Neuron(object):

    def __init__(self, prev = 0):
        self.param = np.zeros(prev)
        if not 0 == prev:
            self.weights = np.ones(prev)

            print("Muestra pesos de la neurona")
            print(self.weights)

# Creamos una red neuronal inicial con n_input y n_output que son el nº de neuronas 
# de la capa de entrada y de la capa de salida

n_input = 5
n_output = 1

n = Network(n_input,n_output)


# Añadimos una capa oculta con el numero de neuronas que queremos
print("\nAñado una capa oculta con 3 neuronas")
n.add_hidden_layer(3)


# Añadimos una capa oculta con el numero de neuronas que queremos
print("\nAñado una capa oculta con 2 neuronas")
n.add_hidden_layer(2)



n.draw_network(.1, .9, .1, .9)
