import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Adiciona o diretório pai ao PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redes_neurais.perceptron import treinar_perceptron, prever as prever_perceptron
from redes_neurais.adaline import treinar_adaline, prever_adaline

# Função para treinar e avaliar um modelo
def avaliar_modelo(modelo, X, Y, max_epocas=1000):
    if modelo == 'perceptron':
        pesos, bias = treinar_perceptron(X, Y, epocas=max_epocas, verbose=False)
        saidas = prever_perceptron(X, pesos, bias)
        acuracia = np.mean(np.array(saidas) == Y) * 100
        return acuracia, max_epocas  # Perceptron não tem early stopping implementado
    else:
        pesos, bias, model = treinar_adaline(X, Y, epocas=max_epocas, verbose=False)
        saidas = prever_adaline(X, pesos, bias, model)
        acuracia = np.mean(np.array(saidas) == Y) * 100
        return acuracia, max_epocas  # Adaline tem early stopping mas vamos simplificar

# Dados das funções lógicas
funcoes = ['AND', 'OR', 'XOR']
X_data = {
    'AND': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'OR': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    'XOR': np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
}
Y_data = {
    'AND': np.array([0, 0, 0, 1]),
    'OR': np.array([0, 1, 1, 1]),
    'XOR': np.array([0, 1, 1, 0])
}

# Avaliar ambos modelos em todas as funções
resultados = {'perceptron': {}, 'adaline': {}}
for funcao in funcoes:
    X, Y = X_data[funcao], Y_data[funcao]
    
    # Avaliar Perceptron
    acuracia, epocas = avaliar_modelo('perceptron', X, Y)
    resultados['perceptron'][funcao] = (acuracia, epocas)
    
    # Avaliar Adaline
    acuracia, epocas = avaliar_modelo('adaline', X, Y)
    resultados['adaline'][funcao] = (acuracia, epocas)

# Configurar o gráfico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de Acurácia
cores = {'perceptron': 'blue', 'adaline': 'orange'}
bar_width = 0.35
index = np.arange(len(funcoes))

for i, modelo in enumerate(resultados):
    acuracias = [resultados[modelo][funcao][0] for funcao in funcoes]
    ax1.bar(index + i * bar_width, acuracias, bar_width, 
            label=modelo.capitalize(), color=cores[modelo])
    
    # Adicionar valores nas barras
    for j, acc in enumerate(acuracias):
        ax1.text(index[j] + i * bar_width, acc + 1, f'{acc:.0f}%', 
                ha='center', va='bottom')

ax1.set_title('Acurácia por Função Lógica e Modelo')
ax1.set_xlabel('Função Lógica')
ax1.set_ylabel('Acurácia (%)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(funcoes)
ax1.set_ylim(0, 120)
ax1.legend()
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Gráfico de Épocas (simplificado - usando valores fixos para demonstração)
epocas_perceptron = [50, 50, 1000]  # Valores ilustrativos
epocas_adaline = [30, 30, 1000]     # Valores ilustrativos

ax2.bar(index, epocas_perceptron, bar_width, label='Perceptron', color='blue')
ax2.bar(index + bar_width, epocas_adaline, bar_width, label='Adaline', color='orange')

# Adicionar valores nas barras
for j, ep in enumerate(epocas_perceptron):
    ax2.text(index[j], ep + 20, f'{ep}', ha='center', va='bottom')
for j, ep in enumerate(epocas_adaline):
    ax2.text(index[j] + bar_width, ep + 20, f'{ep}', ha='center', va='bottom')

ax2.set_title('Épocas para Convergência (Ilustrativo)')
ax2.set_xlabel('Função Lógica')
ax2.set_ylabel('Número de Épocas')
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(funcoes)
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()