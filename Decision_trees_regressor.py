import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Carregando o dataset Iris
arquivo = pd.read_csv('Admission_Predict.csv')
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ',axis=1)

kFold = KFold(n_splits=5,shuffle=True,random_state=7)

modelo = DecisionTreeRegressor()
resultado = cross_val_score(modelo,x,y,cv=kFold)
print('Coeficiente de determinação R2:',resultado.mean())