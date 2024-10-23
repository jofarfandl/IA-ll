import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from perceptron import Perceptron

matplotlib.use('TkAgg')

def draw_2d(title, X, Y, model, xlabel = r'$x_1$', ylabel = r'$x_2$'):
    plt.title(title)
    plt.grid(True)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    _, p = X.shape
    for i in range(p):
        if Y[i] == 0:
            plt.plot(X[0,i], X[1,i], 'or')
        else:
            plt.plot(X[0,i], X[1,i], 'ob')

    w1, w2, b = model.w[0], model.w[1], model.b
    li, ls = -2, 2
    plt.plot(
        [li, ls],
        [(1/w2) * (-w1*(li)-b), (1/w2)*(-w1*(ls)-b)],
        '--k'
    )


def main():
    neuron = Perceptron(2, 0.1)
    X = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])

    # Compuertas

    # Compuerta AND
    Y = np.array([0, 0, 0, 1])

    neuron.fit(X, Y)
    draw_2d('Compuerta AND', X, Y, neuron)
    plt.savefig('Compuerta AND') # plt.show()
    plt.cla()

    # Compuerta OR
    Y = np.array([0, 1, 1, 1])
    neuron.fit(X, Y)
    draw_2d('Compuerta OR', X, Y, neuron)
    plt.savefig('Compuerta OR') # plt.show()
    plt.cla()
    
    # Compuerta XOR
    Y = np.array([0, 1, 1, 0])
    neuron.fit(X, Y)
    draw_2d('Compuerta XOR', X, Y, neuron)
    plt.savefig('Compuerta XOR') # plt.show()
    plt.cla()

    # Sobrepeso

    # X = [Weights, Heights]
    X = np.array([
        np.random.default_rng().uniform(50, 150, 50), # Weight KGs
        np.random.default_rng().uniform(1.5, 2.3, 50) # Heigth Ms
    ])
    Y = (X[0] / X[1]**2) >= 25

    # Normalize weights (Min-max)
    X[0] = (X[0] - X[0].min()) / (X[0].max() - X[0].min())

    # Normalize Heights (Min-max)
    X[1] = (X[1] - X[1].min()) / (X[1].max() - X[1].min())
    
    neuron.fit(X, Y, 250)
    draw_2d('Sobrepeso', X, Y, neuron, 'Peso', 'Altura')
    plt.savefig('Sobrepeso')


if __name__ == '__main__':
    main() 