from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names]) 
y = pd.Series(dados.target)

X_traino, X_teste, Y_traino, Y_teste = train_test_split(x,y,test_size=0.3,random_state=9)

modelo = LogisticRegression(solver='liblinear',C=95, penalty='l1')
modelo.fit(X_traino,Y_traino)
resultado = modelo.score(X_teste,Y_teste)
print('Acuracia: ', resultado)

predicao = modelo.predict(X_teste)

matriz = confusion_matrix(Y_teste,predicao)
print(matriz)