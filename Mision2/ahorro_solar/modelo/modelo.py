import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
data ={
    'ubicacion':['zona_1','zona_2','zona_1','zona_3','zona_2'],
    'tamano_hogar':[3,4,2,5,3],
    'costo_instalacion':[5000,6000,4500,7000,5200],
    'energia_generada':[3000,3200,2500,4000,2900],
    'ahorro_real':[1800,2000,1500,2300,1750]
}
df =pd.DataFrame(data)
# separar variables
x= df.drop(columns='ahorro_real')
y= df['ahorro_real']
# codificar variables categóricas
x_encoded=pd.get_dummies(x)
#print(x_encoded)
# Dividir datos
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2,random_state=42)
# Entrenar modelo
modelo=LinearRegression()
modelo.fit(x_train,y_train)
# Guardar modelo
with open('modelo.pkl','wb') as f:
    pickle.dump(modelo,f)
with open('columnas.pkl','wb') as f:
    pickle.dump(x_encoded.columns.tolist(),f)