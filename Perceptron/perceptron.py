import numpy as np

class Perceptron:
    def __init__(self, n_input, learning_rate):
        self.w = -1 + np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict(self, X):
        p = X.shape[1]
        Y_est = np.zeros(p)
        for i in range(p):
            Y_est[i] = np.dot(self.w, X[:,i]) + self.b
            if Y_est[i] >= 0:
                Y_est[i] = 1
            else:
                Y_est[i] = 0
        return Y_est
    
    def fit(self, X, Y, epochs=50):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                Y_est = self.predict(X[:,i].reshape(-1, 1))
                self.w += self.eta * (Y[i] - Y_est) * X[:,i]
                self.b += self.eta * (Y[i] - Y_est)
