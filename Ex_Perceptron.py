# Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Gerar conjunto de dados
np.random.seed(0)
X = np.random.randn(100, 2)
print(X)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
print(y)
# Função de treinamento do Perceptron
def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0
    for _ in range(epochs):
        for xi, target in zip(X, y):
            update = learning_rate * (target - predict(xi, weights, bias))
            weights += update * xi
            bias += update
    return weights, bias

# Função de previsão
def predict(x, weights, bias):
    return np.where(np.dot(x, weights) + bias >= 0, 1, -1)

# Treinar o Perceptron
weights, bias = train_perceptron(X, y)

# Visualizar os resultados
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
x_min, x_max = plt.xlim()
y_min, y_max = (-bias - weights[0] * x_min) / weights[1], (-bias - weights[0] * x_max) / weights[1]
plt.plot([x_min, x_max], [y_min, y_max], 'k-')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Limite de Decisão do Perceptron')
plt.show()