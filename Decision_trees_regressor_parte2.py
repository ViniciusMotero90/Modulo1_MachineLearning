import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Carregando o dataset Iris
arquivo = pd.read_csv('Admission_Predict.csv')
arquivo.drop('Serial No.',axis=1,inplace=True)
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ',axis=1)

minimo_split = np.array([2, 3, 4, 5, 6, 7])
maximo_nivel = np.array([3, 4, 5, 6, 7, 9, 11])
algoritmo = ['mse', 'friedman_mse','mae']
valores_grid = {'min_samples_split': minimo_split,'max_depth': maximo_nivel,'criterion': algoritmo}


modelo = DecisionTreeRegressor()

gridDecisionTree = GridSearchCV(estimator=modelo,param_grid=valores_grid,cv=5)
gridDecisionTree.fit(x,y)
print('Minimo split:',gridDecisionTree.best_estimator_.min_samples_split)
print('Máximo split:',gridDecisionTree.best_estimator_.max_depth)
print('Algoritmo escolhido:',gridDecisionTree.best_estimator_.criterion)
print('Coeficiente R2:',gridDecisionTree.best_score_)