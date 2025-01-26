from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Configuração para exibir todas as colunas
pd.set_option('display.max_columns', 30)

# Carregar os dados
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=dados.feature_names)  # Corrigido
y = pd.Series(dados.target)

# Dividir os dados em treino e teste
X_traino, X_teste, Y_traino, Y_teste = train_test_split(x, y, test_size=0.3, random_state=9)

# Criar e treinar o modelo
modelo = LogisticRegression(solver='liblinear', C=95, penalty='l1')
modelo.fit(X_traino, Y_traino)

# Avaliar a acurácia
resultado = modelo.score(X_teste, Y_teste)
print('Acurácia: ', resultado)

# Obter as predições de probabilidade
predicao = modelo.predict_proba(X_teste)

# Probabilidade da classe positiva (1)
probs = predicao[:, 1]

# Calcular FPR, TPR e thresholds
fpr, tpr, thresholds = roc_curve(Y_teste, probs)
print('FPR (Taxa de Falsos Positivos): ', fpr)
print('TPR (Taxa de Verdadeiros Positivos): ', tpr)
print('Thresholds (Limiar de Decisão): ', thresholds)

plt.scatter(fpr,tpr)
plt.show()

print(roc_auc_score(Y_teste,probs))