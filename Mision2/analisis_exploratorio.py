import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
1.Generar un dataframe con los datos del fichero titanic.csv
2.Hacer el tratamiento de los nulos
3.Mostrar por pantalla las dimensiones del dataframe, 
el número de datos que contiene,los nombres de sus columnas y filas,
los tipos de datos de las columnas, las 10 primeras filas y las 10 últimas
filas
4.Mostrar por pantalla los datos del pasajero con identificador 148
5.Mostrar por pantalla las filas pares del dataframe
6.Mostrar por pantalla los nombres de las personas que iban en primera clase
  ordenadas alfabeticamente
7.Mostrar por pantalla el porcentaje de personas que sobrevivieron 
  en cada clase
8.Mostrar por pantalla la edad media de las muejeres que viajaban en
  cada clase
9.Añadir una nueva columna boleana para saber si el pasajero era menor de edad o no
10. Mostrar por pantalla el porcentaje de mayores y menores de edad que sobrevivieron
    en cada clase
11. Mostrar por pantalla el porcentaje de mayores de edad que sobrevivieron en cada
    clase
"""




titanic=pd.read_csv('Mision2/titanic.csv')
#imprimo las primeras 5 filas
print(titanic.head())

#Imprimimos las columnas
print(titanic.columns)

#Imprimimos la suma de nulos por columna
print(titanic.isnull().sum())


#Tratamiento de nulos
#1. Rellenar los valores de la edad(Age) con la media
if 'Age' in titanic.columns:
    titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


#Imprimimos nuevamente la suma de nulos por columna
print(titanic.isnull().sum())

#2. como la columna cabin tiene muchos nulos, validamos si tiene
#más de la mitad de valores nulos y si es verdad eliminar la columna
if 'Cabin' in titanic.columns and titanic['Cabin'].isnull().sum()/len(titanic)>0.5:
    #Borramos la columna
    titanic.drop('Cabin',axis=1,inplace=True)


#Imprimimos nuevamente la suma de nulos por columna
print(titanic.isnull().sum())


#3 Rellenar los nulos de puerto de embarque(Embarked) con la moda
if 'Embarked' in titanic.columns:
    titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)


#Imprimimos nuevamente la suma de nulos por columna
print(titanic.isnull().sum())

#Histograma de las edades
plt.figure(figsize=(8,5))
sns.histplot(titanic['Age'],bins=30,color='skyblue')
plt.title("Frecuencia de la edad de los pasajeros")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

#Diagrama de cajas de la edad por la clase
plt.figure(figsize=(8,5))
sns.boxplot(x='Pclass',y='Age',data=titanic,palette='Set2')
plt.title("Distribución de laq edad por clase")
plt.xlabel("Clase")
plt.ylabel("Edad")
plt.show()

#Diagrama de correlación seleccionando solo las columnas numéricas
df_numerico=titanic.select_dtypes(include='number')
correlacion_matriz=df_numerico.corr()
sns.heatmap(correlacion_matriz, annot=True, cmap='coolwarm')
plt.title("Correlación entre las características")
plt.show()
#gfhf

#Diagrama de sectores de fallecidos y supervivientes
plt.figure()
titanic.Survived.value_counts().plot(kind="pie",
labels=["Muertos","Supervivientes"],title='Diagrama de torta de % de vivos y muertos',
autopct='%1.1f%%')
plt.show()

#Diagrama de barras con el número de personas de cada clase
titanic.Pclass.value_counts().plot(kind='bar',title='Número de personas por clase')
plt.show()

print('Dimensiones:',titanic.shape)
print('Número de elementos:',titanic.size)
print('Nombre de filas:',titanic.index)
print('Tipos de datos:',titanic.dtypes)
print('primeras 10 filas:',titanic.head(10))
print('ultimas 10 filas:',titanic.tail(10))

#pasajero 148
print(f"el pasajero 148 es {titanic.loc[titanic['PassengerId']==148]['Name']}")

#Mostrar por pantalla las filas pares del dataframe
print(titanic.iloc[range(0,titanic.shape[0],2)])

#Mostrar los nombres de las personas que iban en primera clase ordenadas alfabeticamente
print(titanic.loc[titanic['Pclass']==1]['Name'].sort_values())

#Mostrar el % de personas que sobrevivieron y murieron
print(titanic['Survived'].value_counts()/titanic['Survived'].count()*100)

#Mostrar el % de personas que sobrevivieron en cada clase
print(titanic.groupby('Pclass')['Survived'].value_counts(normalize=True)*100)

#Mostrar la edad media de las mujeres que viajaban en cada clase
print(titanic.groupby(['Pclass','Sex'])['Age'].mean().unstack()['female'])


#Añadir un nueva columna boleana para ver si el pasajero era menor de edad o no
titanic['es_menor']=titanic['Age']<18

#Mostrar el % de menores y mayores de edad que sobrevivieron en cada clase
print(titanic.groupby(['Pclass','es_menor'])['Survived'].value_counts(normalize=True)*100)