from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Carregando o dataset Iris
iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Treinando o modelo
modelo = DecisionTreeClassifier()
modelo.fit(x, y)

# Plotando a árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(modelo, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
