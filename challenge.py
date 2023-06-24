# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('datos_de_prueba.csv')

# PASOO 1

# Especifica la ruta del archivo CSV
ruta_archivo = 'datos_de_prueba.csv'

# Lee el archivo CSV y crea el DataFrame
df = pd.read_csv(ruta_archivo)

# PASOO 2


# Ahora puedes trabajar con el DataFrame como desees
# Por ejemplo, puedes imprimir las primeras filas para verificar que se haya cargado correctamente
print("HEAD")
print(df.head())
print("INFO")
print(df.info())
print("DESCRIBE")
print(df.describe())
print("COLUMNS")
print(df.columns)
print("SHAPE")
print(df.shape)
print("DTYPES")
print(df.dtypes)
print("#################################################################################")


# PASOO 3

# Eliminar filas con valores nulos
df = df.dropna()
# Eliminar columnas con valores nulos
df = df.dropna(axis=1)
# Convertir una columna a tipo fecha
df['Fecha de inicio de la póliza'] = pd.to_datetime(df['Fecha de inicio de la póliza'])


# PASOO 4
# Create the first plot in Figure 1
plt.figure(1)
df['Edad'].hist()
plt.ylabel('frecuencia relativa')
plt.xlabel('Edad')
plt.title('Histograma1')

# Create a new figure (Figure 2)
plt.figure(2)

# Plot the second plot in Figure 2
df['Puntuación de crédito'].hist()
plt.ylabel('frecuencia relativa')
plt.xlabel('Puntuación de crédito')
plt.title('Histograma2')

# Create a new figure (Figure 3)
plt.figure(3)
# Plot the second plot in Figure 3
#plt.scatter(df['Ingresos'], df['Monto del seguro'])
plt.scatter(df['Edad'], df['Ingresos'])
plt.xlabel('Edad')
plt.ylabel('Ingresos')
plt.title('Gráfico de dispersión')

# Create a new figure (Figure 4)
plt.figure(4)
# Plot the second plot in Figure 4
#plt.scatter(df['Ingresos'], df['Monto del seguro'])
plt.scatter(df['Monto del seguro'], df['Monto del reclamo'])
plt.xlabel('Monto del seguro')
plt.ylabel('Monto del reclamo')
plt.title('Gráfico de dispersión')

# Create a new figure (Figure 5)
plt.figure(5)
# Plot the second plot in Figure 5
plt.scatter(df['Ingresos'], df['Monto del reclamo'])
plt.xlabel('Ingresos')
plt.ylabel('Monto del reclamo')
plt.title('Gráfico de dispersión')

# Create a new figure (Figure 6)
plt.figure(6)
# Plot the second plot in Figure 6
plt.scatter(df['Edad'], df['Monto del reclamo'])
plt.xlabel('Edad')
plt.ylabel('Monto del reclamo')
plt.title('Gráfico de dispersión')

# Display the first figure with its plot
plt.figure(1)



# PASOO 5


# Ingeniería de características
# Convertir variables categóricas en numéricas usando LabelEncoder
le = LabelEncoder()
df['Sexo'] = le.fit_transform(df['Sexo'])
df['Tipo de hogar'] = le.fit_transform(df['Tipo de hogar'])
df['Estado civil'] = le.fit_transform(df['Estado civil'])
df['Tipo de trabajo'] = le.fit_transform(df['Tipo de trabajo'])
df['Educación'] = le.fit_transform(df['Educación'])
df['Tipo de seguro'] = le.fit_transform(df['Tipo de seguro'])

# Convierte la columna 'Fecha del reclamo' al tipo de dato de fecha
df['Fecha del reclamo'] = pd.to_datetime(df['Fecha del reclamo'])
# Aplica el formato 'yyyymmdd' a la columna 'Fecha del reclamo'
df['Fecha del reclamo'] = df['Fecha del reclamo'].dt.strftime('%Y%m%d').astype(int)

# Convierte la columna 'Fecha del reclamo' al tipo de dato de fecha
df['Fecha de inicio de la póliza'] = pd.to_datetime(df['Fecha de inicio de la póliza'])
# Aplica el formato 'yyyymmdd' a la columna 'Fecha del reclamo'
df['Fecha de inicio de la póliza'] = df['Fecha de inicio de la póliza'].dt.strftime('%Y%m%d').astype(int)

# *Crear una columna que represente si un cliente ha presentado un reclamo o no en el pasado
# Se está considerando que como reclámo válido al realizado luego de comenzar la póliza de seguro, los otros casos
# se consideran como que el cliente no ha reclamado.
df['HaPresentadoReclamo'] = df['Fecha del reclamo'] > df['Fecha de inicio de la póliza']

# Crear el modelo
# Seleccionar las características para el modelo y la variable objetivo
y = df['HaPresentadoReclamo']
X = df.drop(['ID del cliente', 'Fecha de inicio de la póliza', 'Fecha del reclamo', 'Monto del reclamo', 'HaPresentadoReclamo'], axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo de regresión logística
model_logitic = LogisticRegression()
model_logitic.fit(X_train, y_train)
# Entrenar el modelo de árbol de decisión
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)

# Hacer predicciones
y_logitic_pred = model_logitic.predict(X_test)
y_tree_pred = model_tree.predict(X_test)

# Evaluar los modelos con datos nuevos no vistos (datos de testeo)
metrics = {'Accuracy': accuracy_score, 'Recall': recall_score, 'Precision': precision_score, 'F1': f1_score}
for name, metric in metrics.items():
    print(f"{name} of Logistic Regression:", metric(y_test, y_logitic_pred))
    print(f"{name} of Decision Tree Classifier:", metric(y_test, y_tree_pred))

# Comparar los modelos
# Aquí se asume que un modelo es "mejor" si tiene un mejor puntaje F1. Esto se debe a que el puntaje F1 es una media armónica de la precisión y el recall, por lo que es una buena medida para comparar modelos. Si necesitas maximizar una métrica específica (por ejemplo, la precisión o el recall), deberías utilizar esa métrica para la comparación en lugar del puntaje F1.
if f1_score(y_test, y_logitic_pred) > f1_score(y_test, y_tree_pred):
    best_model = model_logitic
    print("The Logistic Regression model is the best model.")
else:
    best_model = model_tree
    print("The Decision Tree Classifier model is the best model.")



plt.show()