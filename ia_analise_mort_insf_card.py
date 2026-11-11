import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Carregar os dados
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
print(df.info())  # Verificar informações gerais do dataset
print(df.head())  # Mostrar as primeiras linhas

# Pré-processamento: converter variáveis categóricas em dummies
df_getdummy = pd.get_dummies(df, columns=['anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                                          'high_blood_pressure', 'platelets', 'serum_creatinine',
                                          'serum_sodium', 'sex', 'smoking', 'time'])
print(df_getdummy.info())  # Verificar a nova estrutura do dataframe

# Separação em variáveis independentes (X) e variável alvo (y)
X = df_getdummy.drop("DEATH_EVENT", axis=1)
y = df_getdummy["DEATH_EVENT"]

# Divisão dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=181)

# Criar e treinar o classificador de árvore de decisão
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dtree.fit(X_train, y_train)

# Fazer previsões
predictions = dtree.predict(X_test)

# Visualizar a árvore de decisão
fig = plt.figure(figsize=(27, 13))
plot_tree(dtree, feature_names=X.columns, fontsize=12, filled=True, class_names=["MORTE(não)", "MORTE(sim)"])
plt.show()
