import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 30)

# Carregar os dados
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=dados.feature_names)  # Corrigido
y = pd.Series(dados.target)

normalizador = MinMaxScaler(feature_range=(0,1))
X_norm = normalizador.fit_transform(x)

X_traino, X_teste, Y_treino, Y_teste = train_test_split(X_norm,y,test_size=0.3,random_state=16)
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_traino,Y_treino)

resultado = modelo.score(X_teste,Y_teste)
print("Acurácia: ",resultado)