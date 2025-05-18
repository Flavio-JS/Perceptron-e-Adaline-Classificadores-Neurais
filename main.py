import numpy as np
from redes_neurais.perceptron import treinar_perceptron, prever as prever_perceptron
from redes_neurais.adaline import treinar_adaline, prever_adaline

def get_dataset(nome_funcao):
    if nome_funcao == 'AND':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 0, 0, 1])
    elif nome_funcao == 'OR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 1, 1, 1])
    elif nome_funcao == 'XOR':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        Y = np.array([0, 1, 1, 0])
    else:
        raise ValueError("Função lógica inválida.")
    return X, Y

def main():
    print("=== Classificador Neural: Perceptron e Adaline ===\n")

    # Opções com números
    opcoes_funcoes = {
        '1': 'AND',
        '2': 'OR',
        '3': 'XOR'
    }

    opcoes_modelos = {
        '1': 'perceptron',
        '2': 'adaline'
    }

    # Escolha da função lógica
    print("Escolha a função lógica:")
    for k, v in opcoes_funcoes.items():
        print(f"{k} - {v}")

    while True:
        escolha_funcao = input("Digite o número correspondente (1, 2 ou 3): ").strip()
        if escolha_funcao in opcoes_funcoes:
            funcao = opcoes_funcoes[escolha_funcao]
            if funcao == 'XOR':
                print("\nAVISO: XOR não pode ser resolvido com um único Perceptron ou Adaline.")
                print("Será mostrado o resultado, mas não será capaz de aprender corretamente.\n")
            break
        print("Opção inválida. Tente novamente.")

    # Escolha do modelo
    print("\nEscolha o modelo de rede neural:")
    for k, v in opcoes_modelos.items():
        print(f"{k} - {v.capitalize()}")

    while True:
        escolha_modelo = input("Digite o número correspondente (1 ou 2): ").strip()
        if escolha_modelo in opcoes_modelos:
            modelo = opcoes_modelos[escolha_modelo]
            break
        print("Opção inválida. Tente novamente.")

    X, Y = get_dataset(funcao)

    verbose = input("\nMostrar detalhes do treinamento? (s/n): ").strip().lower() == 's'

    if modelo == 'perceptron':
        pesos, bias = treinar_perceptron(X, Y, verbose=verbose)
        saidas = prever_perceptron(X, pesos, bias)
    elif modelo == 'adaline':
        pesos, bias, model = treinar_adaline(X, Y, verbose=verbose)
        saidas = prever_adaline(X, pesos, bias, model)

    # Cálculo da acurácia
    acuracia = np.mean(np.array(saidas) == Y) * 100

    print(f"\n--- Resultado Final - Função: {funcao} - Modelo: {modelo.capitalize()} ---")
    for entrada, saida, esperado in zip(X, saidas, Y):
        print(f"Entrada: {entrada}, Saída prevista: {saida}, Esperado: {esperado}")
    print(f"\nAcurácia: {acuracia:.2f}%")

    if funcao == 'XOR' and acuracia < 100:
        print("\nComo esperado, o modelo não conseguiu aprender a função XOR, que requer uma rede multicamadas.")

if __name__ == "__main__":
    main()
