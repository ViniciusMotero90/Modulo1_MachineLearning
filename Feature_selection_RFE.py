from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
import pandas as pd

# Configurações do pandas
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 320)

# Carregar os dados
arquivo = pd.read_csv('Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)

# Separar X (variáveis independentes) e y (variável dependente)
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

# Modelo Decision Tree para Regressão
modelo = DecisionTreeRegressor()

# RFE com n_features_to_select=5
rfe = RFE(estimator=modelo, n_features_to_select=5)
fit = rfe.fit(x, y)

# Exibir resultados
print('Número de atributos selecionados:', fit.n_features_)
print('Atributos selecionados:', fit.support_)
print('Ranking dos atributos:', fit.ranking_)
