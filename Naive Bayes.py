from sklearn.datasets import load_iris
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

X_traino, X_teste, Y_treino, Y_teste = train_test_split(x,y,test_size=0.3,random_state=67)

modelo = GaussianNB()
modelo.fit(X_traino,Y_treino)

resultado = modelo.score(X_teste,Y_teste)
print('Acurácia: ',resultado)