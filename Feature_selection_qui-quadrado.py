from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
import pandas as pd

x = [[12,2,30],[15,11,6],[16,8,90],[5,3,20],[4,14,5],[2,5,70]]
y = [1,1,1,0,0,0]

algoritmo = SelectKBest(score_func=chi2,k=2)
dados_das_melhores_preditores = algoritmo.fit_transform(x,y)

print('Score: ',algoritmo.scores_)
print('Resultado das transformações:\n',dados_das_melhores_preditores)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=[iris.feature_names])
Y = pd.Series(iris.target)

algoritmo = SelectKBest(score_func=chi2,k=2)
dados_das_melhores_preditores = algoritmo.fit_transform(X,Y)

print('Score: ',algoritmo.scores_)
print('Resultado das transformações:\n',dados_das_melhores_preditores)