from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = pd.DataFrame(iris.data, columns=[iris.feature_names])
Y = pd.Series(iris.target)

algoritmo = SelectKBest(score_func=f_classif,k=2)
dados_das_melhores_preditores = algoritmo.fit_transform(X,Y)

print('Score: ',algoritmo.scores_)
print('Resultado das transformações:\n',dados_das_melhores_preditores)