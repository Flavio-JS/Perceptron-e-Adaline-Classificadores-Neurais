import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Importando os modelos
from redes_neurais.perceptron import treinar_perceptron, prever as prever_perceptron
from redes_neurais.adaline import treinar_adaline, prever_adaline

def plot_decision_regions(X, y, model_type, ax, title):
    """Plota as regiões de decisão para um modelo."""
    
    # Define os limites do gráfico
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Cria uma malha de pontos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Treina o modelo
    if model_type == 'perceptron':
        pesos, bias = treinar_perceptron(X, y, verbose=False)
        Z = np.array([degrau(np.dot(np.array([xi1, xi2]), pesos) + bias) 
                     for xi1, xi2 in zip(xx.ravel(), yy.ravel())])
    else:
        pesos, bias, model = treinar_adaline(X, y, verbose=False)
        Z = np.array([degrau(np.dot(np.array([xi1, xi2]), pesos) + bias) 
                     for xi1, xi2 in zip(xx.ravel(), yy.ravel())])
    
    Z = Z.reshape(xx.shape)
    
    # Define cores personalizadas
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA'])
    
    # Plota as regiões de decisão
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    
    # Plota os pontos de dados
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, label=f'Classe {cl}',
                   edgecolor='black')
    
    ax.legend()
    ax.grid(True)

def degrau(u):
    """Função degrau para compatibilidade."""
    return 1 if u >= 0 else 0

# Dados das funções lógicas
logic_functions = {
    'AND': {
        'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'y': np.array([0, 0, 0, 1])
    },
    'OR': {
        'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'y': np.array([0, 1, 1, 1])
    },
    'XOR': {
        'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        'y': np.array([0, 1, 1, 0])
    }
}

# Configuração da figura
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparação entre Perceptron e Adaline em Funções Lógicas', fontsize=16)

# Gerar gráficos para cada função lógica e modelo
for i, (name, data) in enumerate(logic_functions.items()):
    # Perceptron
    plot_decision_regions(data['X'], data['y'], 'perceptron', axes[0, i], 
                         f'Perceptron - {name}')
    
    # Adaline
    plot_decision_regions(data['X'], data['y'], 'adaline', axes[1, i], 
                         f'Adaline - {name}')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()