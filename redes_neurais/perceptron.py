import numpy as np
from .utils import degrau

def treinar_perceptron(X, y, taxa_aprendizado=0.1, epocas=10, verbose=False):
    pesos = np.zeros(X.shape[1])
    bias = 0
    if verbose:
        print("\nTreinando Perceptron...\n")
    
    for epoca in range(epocas):
        if verbose:
            print(f"Época {epoca + 1}")
        
        for xi, target in zip(X, y):
            u = np.dot(xi, pesos) + bias
            saida = degrau(u)
            erro = target - saida
            pesos += taxa_aprendizado * erro * xi
            bias += taxa_aprendizado * erro
            
            if verbose:
                print(f"Entrada: {xi}, Esperado: {target}, Saída: {saida}, Erro: {erro}, Pesos: {pesos}, Bias: {bias}")
        
        if verbose:
            print("-" * 50)
    
    return pesos, bias

def prever(X, pesos, bias):
    return [degrau(np.dot(xi, pesos) + bias) for xi in X]
