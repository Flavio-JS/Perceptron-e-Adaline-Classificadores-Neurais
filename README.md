# Perceptron e Adaline - Classificadores Neurais

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementação dos algoritmos Perceptron e Adaline para classificação de funções lógicas básicas.

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes Python)

## 🛠 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/Flavio-JS/Perceptron-e-Adaline-Classificadores-Neurais.git
cd perceptron-and-adaline
```

2.Instale as dependências:

```bash
pip install -r requirements.txt
```

## 🚀 Como Executar

1. Interface Principal
   Para usar a interface interativa:

```bash
python main.py
```

Siga as instruções no terminal para:

Selecionar a função lógica (AND, OR, XOR)

Escolher o modelo (Perceptron ou Adaline)

Definir se deseja ver detalhes do treinamento

2. Gerar Gráficos Comparativos
   Para gerar os gráficos de desempenho:

```bash
python -m scripts.grafico-de-comparacao
```

3. Gerar Gráficos de Funções Lógicas com Perceptron e Adaline

```bash
python -m scripts.plot_logic_functions
```

## 🧠 Sobre os Modelos

### Perceptron
- **Algoritmo** de aprendizado supervisionado
- **Aprendizado online** (atualiza pesos a cada exemplo)
- Adequado para **problemas linearmente separáveis**

### Adaline (Adaptive Linear Neuron)
- Versão aprimorada do Perceptron
- Usa **função de ativação linear** durante o treinamento
- Implementa **momento adaptativo** para convergência mais rápida

## 📊 Resultados Esperados

| Função | Perceptron | Adaline |
|--------|-----------|---------|
| AND    | 100%      | 100%    |
| OR     | 100%      | 100%    |
| XOR    | 50%       | 50%     |

> **Nota**: XOR não pode ser resolvido por um único neurônio, servindo apenas como demonstração.