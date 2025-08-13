import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Cargamos los datos
iris=load_iris()
#Convertimos a data frame
df= pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species']= iris.target
df['species']= df['species'].map(dict(zip(range(3),iris.target_names)))

#imprimimos las primeras  5 filas
print(df.head())


#Tipos de datos de las columnas
print(df.info())


#Estadisticas descriptivas
print(df.describe())



#Contar cuantas flores hay de cada clase
especie_count= df['species'].value_counts()
print(especie_count)


#Crear gráfico de barras
plt.bar(especie_count.index,especie_count.values, color='skyblue')
plt.xlabel('Especie')
plt.ylabel('Cantidad')
plt.title('Distribución de especies')
plt.show()

df['species']=iris.target

#Crear la matriz de correlación
correlacion_matriz=df.corr()
sns.heatmap(correlacion_matriz,annot=True,cmap='coolwarm')
plt.title('Correlación entre variables')
plt.show()