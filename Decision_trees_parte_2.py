from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Carregando o dataset Iris
iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Definindo os parâmetros para GridSearchCV
minimo_split = np.array([2, 3, 4, 5, 6, 7, 8])
maximo_nivel = np.array([3, 4, 5, 6])
algoritmo = ['gini', 'entropy']
valores_grid = {
    'min_samples_split': minimo_split,
    'max_depth': maximo_nivel,
    'criterion': algoritmo
}

# Treinando o modelo usando GridSearchCV
modelo = DecisionTreeClassifier()
gridDecisionTree = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
gridDecisionTree.fit(x, y)

# Resultados do melhor modelo
melhor_modelo = gridDecisionTree.best_estimator_
print('Mínimo split:', melhor_modelo.min_samples_split)
print('Máximo profundidade:', melhor_modelo.max_depth)
print('Algoritmo escolhido:', melhor_modelo.criterion)
print('Acurácia:', gridDecisionTree.best_score_)

# Visualizando a árvore de decisão final
plt.figure(figsize=(12, 8))
plot_tree(
    melhor_modelo,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.show()
