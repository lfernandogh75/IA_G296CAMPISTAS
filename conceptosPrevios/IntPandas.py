import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Acceso a los Elementos de un objeto Series
# cada elemento de objeto series tiene un identificador úmico
s=pd.Series([2,4,6,8],index=["num1","num2","num3","num4"])
print(s)
# accediendo al tercer elemento del objeto
print(s["num3"])
# aceder por la posición
print(s.iloc[2])
print(s.loc["num3"])
#acedoendo al segundo y tercer elemento por posición
print(s.iloc[2:4])

# Operaciones aritméticas con series
#sumar
print(f"suma {np.sum(s)}")
print("suma ",np.sum(s))

# creación de un objeto series denominado temperaturas
temperaturas=pd.Series([4.4,5.1,6.1,6.2,6.1,6.1,5.2,4.7,4.1,3.9])
s=pd.Series(temperaturas,name="Temperaturas")
print(s)
s.plot()
#plt.show()

#Creación de un objeto DataFrame
personas={
    "peso":pd.Series([90,80,70,60],["Santiago","Marcelo","Luis","Alejandra"]),
    "altura":pd.Series({"Santiago":180,"Marcelo":172,"Luis":174,"Alejandra":160}),
    "hijos":pd.Series([2,3],["Luis","Marcelo"])
}
df =pd.DataFrame(personas)
print(df)
df2=pd.DataFrame(
    personas,
    columns=["altura","peso"],
    index=["Santiago","Luis","Marcelo"]
)
print(df2)
#Acceso a los elementos
print(df["peso"])
#combinar metodos
df3=(df["peso"]>=60) & (df["peso"]<80)
print(df3)

print(df.iloc[1:3])
#consultas avanzadas
df4=df.query("altura >= 170 and peso >70")
print(df4)
#copiar un dataframe
df_copy =df.copy()
#añadir una nueva columna
df_copy["Cumpleaños"]=[1990,1978,1980,1994]
print(df_copy)
#añadir una columna calculada
df_copy["años"]=2025 - df_copy["Cumpleaños"]
print(df_copy)
#añadir una nueva columna craando un DataFrame nuevo
df_mod=df_copy.assign(mascotas=[1,3,0,0])
print(df_mod)
#Eliminar una columna
del df_mod["peso"]
print(df_mod)

df_copy["fecha_nacimiento"] = 2025 - df_copy["Cumpleaños"]
print(df_copy)