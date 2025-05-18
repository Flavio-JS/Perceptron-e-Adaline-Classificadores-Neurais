import numpy as np
from .utils import degrau

class Adaline:
    def __init__(self):
        self.pesos = None
        self.bias = None
        self.gama = 0.9  # Fator de momento

    def treinar(self, X, Y, epocas=1000, taxa_aprendizado=0.1, verbose=False, paciencia=50):
        # Não normalizamos para problemas simples de portas lógicas
        self.pesos = np.random.randn(X.shape[1]) * 0.1
        self.bias = np.random.randn() * 0.1
        self.momento_pesos = np.zeros(X.shape[1])
        self.momento_bias = 0
        
        if verbose:
            print("\nTreinando Adaline...\n")
            
        melhor_erro = float('inf')
        espera = 0
        
        for epoca in range(epocas):
            erro_total = 0
            for xi, target in zip(X, Y):
                u = np.dot(xi, self.pesos) + self.bias
                erro = target - u
                erro_total += erro ** 2
                
                # Atualização com momento adaptativo
                taxa_efetiva = taxa_aprendizado / (epoca + 1) ** 0.5
                delta_pesos = taxa_efetiva * erro * xi
                delta_bias = taxa_efetiva * erro
                
                self.pesos += delta_pesos + self.gama * self.momento_pesos
                self.bias += delta_bias + self.gama * self.momento_bias
                
                self.momento_pesos = delta_pesos
                self.momento_bias = delta_bias
                
            if verbose:
                print(f"Época {epoca + 1}, Erro Total: {erro_total:.6f}")

            # Early stopping adaptado
            if erro_total < 0.001:  # Limite mais rigoroso
                if verbose:
                    print(f"Convergência alcançada na época {epoca + 1}")
                break
                
            if erro_total >= melhor_erro:
                espera += 1
                if espera >= paciencia:
                    if verbose:
                        print(f"Early stopping na época {epoca+1}")
                    break
            else:
                melhor_erro = erro_total
                espera = 0

    def prever(self, X):
        return [degrau(np.dot(xi, self.pesos) + self.bias) for xi in X]

def treinar_adaline(X, Y, epocas=1000, taxa_aprendizado=0.1, verbose=False):
    model = Adaline()
    model.treinar(X, Y, epocas, taxa_aprendizado, verbose)
    return model.pesos, model.bias, model

def prever_adaline(X, pesos, bias, model):
    # Criamos uma instância temporária para compatibilidade
    temp_model = Adaline()
    temp_model.pesos = pesos
    temp_model.bias = bias
    return temp_model.prever(X)