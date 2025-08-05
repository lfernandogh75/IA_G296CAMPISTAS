# Introdicción Pandas
Python 3.9.2
## librerias
```
import pandas as pd 
```
## Creación de objeto Serie
```
# Creación de objeto Serie
s= pd.Series([2,4,6,8,10])
print(s)

# Creación de un objeto Serie inicializándo con un diccionario de python
altura={"Santigo":180,"Marcelo":172,"Luis":174,"Alejandra":160}
s=pd.Series(altura)
print(s)
"""
Creación de un objeto series inicializándolo con algunos
de los elementos de un diccionario de python
"""
altura={"Santigo":180,"Marcelo":172,"Luis":174,"Alejandra":160}
s=pd.Series(altura,index=["Marcelo","Luis"])
print(s)

#Creación de un objeto Series inicializandolo con un escalar
s=pd.Series(34,["test1","test2","test3"])
print(s)
```