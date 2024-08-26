# Importar bibliotecas
#Numpy para cálculos vetoriais
#Matplotlib para plotar gráficos
import numpy as np
import matplotlib.pyplot as plt

#Define a semente aleatória, que garantirá a repdução posterior dos resultados
np.random.seed(0)

#Gera 100 amostras de 2 propriedades
X = np.random.randn(100, 2)

#Verifica se a soma dos elementos da colunas de X é maior que 0, se verdadeiro atribui 1, se falso, atribui 0
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Função de treinamento do Perceptron
def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    """
    Treina um modelo Perceptron nos dados fornecidos.

    Parâmetros:
        X : As características de entrada.
        y : As labels de saída.
        learning_rate: A taxa de aprendizado do Perceptron. Padrão é 0.1.
        epochs: O número de épocas para treinar o Perceptron. Padrão é 10.

    Retorna:
        Os pesos e o Viés treinados do Perceptron.
    """
    #Inicializa os pesos e vieses em zero
    weights = np.zeros(X.shape[1])
    bias = 0
    
    #Itera treinando o modelo a cada época
    for _ in range(epochs):
        #Itera sobre cada valor de X e y, calcula a atualização, e ajusta o peso e o viés
        
        for xi, target in zip(X, y):
            
            #Cada atualização é feita sobre a diferença da projeção para o valor real, 
            #ponderados pela taxa de aprendizagem
            update = learning_rate * (target - predict(xi, weights, bias))
            
            #O peso recebe a soma acumulada de si mesmo mais uma atualização ponderada por xi
            weights += update * xi
            
            #O viés recebe a soma acumulada de si mesmo mais a atualização 
            bias += update
            
    return weights, bias
    #Devolve o peso e o viés calculados para o modelo

# Função de previsão
def predict(x, weights, bias):
    """
    Realiza a predição para o valores de y, através da entre dos valores de X, dos pesos, e vieses.

    Parâmetros:
        X : As características de entrada.
        weights: Pesos calculados pela função de treino
        bias: Vieses calculados pela função de treino
    """
    return np.where(np.dot(x, weights) + bias >= 0, 1, -1)

# Treinar o Perceptron usando a função de treino
weights, bias = train_perceptron(X, y)

# Visualizar os resultados
#Definição das variáveis a serem plotadas
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

#Definindo Limites de X
x_min, x_max = plt.xlim()

#Definindo Limites de y
y_min, y_max = (-bias - weights[0] * x_min) / weights[1], (-bias - weights[0] * x_max) / weights[1]

#Plotando dados de X e y
plt.plot([x_min, x_max], [y_min, y_max], 'k-')

#Adicionando nome dos eixos e títulos
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Limite de Decisão do Perceptron')

#Mostrando plot
plt.show()