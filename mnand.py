import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.tanh(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        error = self.a2 - y
        delta2 = error * (1 - np.tanh(self.z2)**2)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = np.dot(delta2, self.W2.T) * (1 - np.tanh(self.z1)**2)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((output - y)**2)
                print(f"Epoch {epoch}: loss = {loss:.4f}")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1], [1], [1], [0]])
mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.train(X, y, learning_rate=0.1, epochs=1000)
y_pred = mlp.forward(X)
print("Predicted output:")
print(y_pred)