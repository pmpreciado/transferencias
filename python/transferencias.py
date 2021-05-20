# -*- coding: utf-8 -*-
"""
Máster en Inteligencia Artificial
Detección de transferencias fraudulentas
UCAV

Este script desarrolla un modelo de clasificación basado en Random Forest
para detectar transferencias fraudulentas.

Pedro Mª Preciado
19/02/2021
"""



#%% Importación de las bibliotecas utilizadas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#%% Importación del dataset con las transferencias
# El CSV tiene estas características:
#   - Títulos de las columnas
#   - Carácter separador de columnas: ","
#   - Carácter separador de filas: "\r\n" (DOS)
#   - Valores entrecomillados
#   - Codificación UTF-8
ruta_dataset = r"dataset\htresepa.csv";
dtypes_dicc = {"pob_ordnte": "str", "prov_ordnte": "str", "prov_benef": "str"}
dataset = pd.read_csv(ruta_dataset, dtype = dtypes_dicc)
dataset.fillna("", inplace = True)


# Si queremos trabajar con un conjunto de datos más pequeño, podemos
# extraer una muestra:
# dataset = dataset.sample(frac=0.01, random_state = 1)

#%% Mostramos las columnas del dataset
print("Columnas del dataset:")
print(dataset.dtypes)

#%% Mostramos las primeras y las últimas filas
print("Primeros y últimos registros:")
print(dataset.info)



#%% Transformamos las categorías tipo texto en numéricas para facilitar
# el desarrollo del modelo
#LabelEncoder - https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.LabelEncoder.html
#OneHotEncoder - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html?highlight=onehotencoder#sklearn.preprocessing.OneHotEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cat_enc = ["identificador", "tipo_pago", "bic_destino", "cod_ordnte", \
           "pob_ordnte", "prov_ordnte", "iban_ordnte", "nom_benef", \
           "dir_benef", "pob_benef", "prov_benef", "pais_benef", \
           "iban_benef", "cat_proposito"]
    
for c in cat_enc:
    print("Transformando característica: {}".format(c))
    le.fit(dataset[c])
    dataset[c] = le.transform(dataset[c])


#%% Información estadística de los datos
print("Información estadística de los datos:")
print(dataset.describe())


#%% Visualizamos la distribución de la fecha de las transferencias
# Usamos matplotlib porque el histograma de seaborn muestra las fechas del
# eje X como números en coma flotante
dataset["fecha_transf_aux"] = \
    pd.to_datetime(dataset["fecha_transf"].astype(str), format='%Y%m%d')
plt.hist(dataset["fecha_transf_aux"], bins = 10, rwidth = 0.7)
dataset.drop(["fecha_transf_aux"], axis=1, inplace=True)
                  


#%% Visualizamos la distribución del importe de las transferencias
# Mostramos cuatro representaciones, por rango de importe:
#    - De 0 en adelante
#    - De 0 a 10000€
#    - De 0 a 1000€
#    - De 100000€ en adelante
fig, axes = plt.subplots(4, figsize=(14, 16))

fig.suptitle("Distribución del importe de las transferencias")
sns.distplot(dataset["importe_euros"], axlabel="a) Importe en euros (0-)", hist=False, label="Densidad", color="red", ax=axes[0])

d1000 = dataset[dataset["importe_euros"] <= 1000]
sns.distplot(d1000["importe_euros"], axlabel="b) Importe en euros (0-1000)", label="Densidad", color="green", ax=axes[1])

d10000 = dataset[dataset["importe_euros"] <= 10000]
sns.distplot(d10000["importe_euros"], axlabel="c) Importe en euros (0-10000)",  label="Densidad", color="blue", ax=axes[2])

dx = dataset[dataset["importe_euros"] >= 100000]
sns.distplot(dx["importe_euros"], axlabel="d) Importe en euros (100000-)", label="Densidad", color="purple", ax=axes[3])





#%% Comparamos el número de transferencias fraudulentas con el de
# transferencias legítimas
ds_leg = dataset[dataset["fraude"] == 0]
ds_fraude = dataset[dataset["fraude"] == 1]
n_leg = len(ds_leg)
n_fraude = len(ds_fraude)
imp_leg = ds_leg["importe_euros"].sum()
imp_fraude = ds_fraude["importe_euros"].sum()
porcentaje = 100 * n_fraude / n_leg
print("Núm. transferencias legítimas: {}".format(n_leg))
print("Núm. transferencias fraude: {}".format(n_fraude))
print("% fraude: {}".format(porcentaje))
print("Importe transferencias legítimas: {}".format(imp_leg))
print("Importe transferencias fraude: {}".format(imp_fraude))

#%% Matriz de correlación
# Generamos una matriz de correlación con los datos de entrada.
# La matriz muestra un mapa de calor que ayuda a entender los datos objeto
# de estudio
# Por ejemplo, se puede observar que la característica tipo_pago está muy
# relacionada con pais_benef. Se explica porque ciertos tipos de pago no 
# están disponibles en todos los países.
matriz_correlacion = dataset.corr()
plt.figure(figsize = (15, 10))
axes = plt.axes()
sns.heatmap(matriz_correlacion, vmax = 0.75, square = True, ax = axes)
axes.set_title("Matriz de correlación")
plt.show()


#%% Hacemos la separación entre la variable dependiente (ojetivo)
# y las independientes (características)
X = dataset.drop(["fraude"], axis=1)    # Características (features)
Y = dataset["fraude"]                   # Objetivo (target)
print(X.shape)
print(Y.shape)

# Creamos un np.ndarray con los valores de los dataset (sin las cabeceras)
X_data = X.values
Y_data = Y.values


#%% Dividimos el dataset entre entrenamiento (75%) y test (25%)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
    train_test_split(X_data, Y_data, test_size = 0.25, random_state = 0)


# %% Escalamos (normalizamos) los atributos
# Se hace para que los campos numéricos se muevan en el el mismo rango de 
# valores
from sklearn.preprocessing import StandardScaler
st_scaler = StandardScaler()
X_train_sc = st_scaler.fit_transform(X_train)
X_test_sc = st_scaler.transform(X_test)


#%% Creamos el clasificador Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(X_train_sc, Y_train)
y_pred = rfc.predict(X_test_sc)


#%% Implementación de otros algoritmos
# Quitar los comentarios para evaluar el deseado
"""
#%%  Prueba del clasificador Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnbc = GaussianNB()
gnbc.fit(X_train_sc, Y_train)
y_pred = gnbc.predict(X_test_sc)


#%%  Prueba del clasificador Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
mnnbc = MultinomialNB()
mnnbc.fit(X_train_sc, Y_train)
y_pred = mnnbc.predict(X_test_sc)

#%%  Prueba del clasificador Categorical Naive Bayes

from sklearn.naive_bayes import CategoricalNB
cnbc = CategoricalNB()
cnbc.fit(X_train_sc, Y_train)
y_pred = cnbc.predict(X_test_sc)

#%%  Prueba del clasificador Bernoulli Naive Bayes

from sklearn.naive_bayes import BernoulliNB
bnbc = BernoulliNB()
bnbc.fit(X_train_sc, Y_train)
y_pred = bnbc.predict(X_test_sc)



#%%  Prueba del clasificador KNN
from sklearn.neighbors import KNeighborsClassifier
n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_sc, Y_train)
y_pred = knn.predict(X_test_sc)

#%%  Prueba del clasificador SVM con kernel 'rbf'

from sklearn import svm
svmc = svm.SVC()
svmc.fit(X_train_sc, Y_train)
y_pred = svmc.predict(X_test_sc)


#%%  Prueba del clasificador basado en redes neuronales

from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier(random_state = 1, max_iter = 300)
mlpc.fit(X_train_sc, Y_train)
mlpc.predict_proba(X_test_sc)
"""

#%% Evaluamos el rendimiento del modelo sobre los datos de test.
#
# Calculamos estos parámetros de rendimiento:
#    - Accuracy 
#    - Precision
#    - Recall
#    - F score
#    - MCC (Coeficiente de correlación de Mathews)
#
# TODO: Interpretar los parámetros
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
        f1_score, matthews_corrcoef

print("Evaluación del modelo")

num_casos_fraude = len(ds_fraude)
num_errores_prediccion = (y_pred != Y_test).sum()

print("Casos de fraude: {}".format(num_casos_fraude))
print("Errores de predicción (falsos positivos + falsos negativos): {}".format(num_errores_prediccion))

accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: {}".format(accuracy))

precision = precision_score(Y_test, y_pred)
print("Precision: {}".format(precision))

recall = recall_score(Y_test, y_pred)
print("Recall: {}".format(recall))

f1score = f1_score(Y_test, y_pred)
print("F1 score: {}".format(f1score))

mcc = matthews_corrcoef(Y_test, y_pred)
print("Coeficiente de correlación de Matthews (MCC): {}".format(mcc))



#%% Generamos una matriz de confusión para visualizar, de manera gráfica,
# el rendimiento del algoritmo
from sklearn.metrics import confusion_matrix

etiquetas = ["Transferencias legítimas", "Transferencias fraudulentas"]
matriz_confusion = confusion_matrix(Y_test, y_pred)
plt.figure(figsize = (10, 10))
sns.heatmap(matriz_confusion, xticklabels = etiquetas, yticklabels = etiquetas, \
            annot = True, fmt = "d");
plt.title("Matriz de confusión")
plt.ylabel("Realidad")
plt.xlabel("Predicción")
plt.show()


#%% Ya por curiosidad, vamos a generar el diagrama de uno de los árboles 
# que componen el Random Forest.
# Puede ayudar a analizar qué decisiones va tomando el algoritmo en cada nodo
# hasta decidir si la transferencia es legítima o fraudulenta

from sklearn.tree import export_graphviz
from IPython.display import Image

# Para instalar el módulo pydot:
#   conda install pydot-ng
#   conda install pydot
#   conda install graphviz
import pydot


# Elegimos uno de los árboles del random forest
indice_arbol = np.random.randint(1, len(rfc.estimators_) - 1)
arbol = rfc.estimators_[indice_arbol]

# Para generar el gráfico nos apoyamos en un fichero DOT
# Los ficheros DOT contienen instrucciones para dibujar gráficos
caracteristicas = list(X.columns)
export_graphviz(arbol, out_file = r"c:\temp\tree.dot", \
                feature_names = caracteristicas, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file(r"c:\temp\tree.dot")

# Creamos una imagen PGN a partir del fichero .dot
from IPython.display import display
grafico_arbol = graph.create_png()
imagen_arbol = Image(grafico_arbol)
display(imagen_arbol)



