import numpy as np
import matplotlib.pyplot as plt
#Matriz de 2 x 4 con 0
a=np.zeros((2,4))

print(a)

#Matriz de 2 x 4 con 1
b=np.ones((2,4))
print(b)

#Imprimir las dimensiones de la matriz
print("Dimensiones: ",a.shape)
#Imprimir el número de dimensiones de la matriz
print("Número de dimensiones: ",a.ndim)
#Imprimir el tamaño de la matriz
print("Tamaño: ",a.size)

"""
Array o matriz cuyos valores son todos el valor 
indicado como segundo parametro de la función
"""
c=np.full((2,3,4),8)
print(c)

"""
Inicializa los valores del array con lo que haya en 
memoria en el momento

El llenado del empty es aleatorio
"""

d=np.empty((2,3,9))
print(d)

#Inicializasción del array usando uno de python
d=np.array([[1,2,3],[4,5,6]])
print(d)
#Imprimimos el tamaño
print("Tamaño",d.shape)

"""
Creación del array utilizando una función basada
en rangos

(mínimo, maximo, número de elementos del array)
"""

print(np.linspace(0,6,10))

#Inicialización del array con valores aleatorios
e=np.random.rand(2,3,4)
print(e)

"""
Inicialización del array con valores aleatorios 
"""
f=np.random.rand(2,4)
print(f)

"""
Histograma con valores aleatorios
"""

g=np.random.rand(100)

plt.hist(g,bins=100)
plt.show()

"""
Histograma con valores personalizados
"""
h=np.array([1,2,3,2,2,2,4,5,6,7,8])

plt.hist(h,bins=16)
plt.show()

#Inicialización de un array/matriz usando una función
def func(x,y):
    return x+2*y

i=np.fromfunction(func, (3,5))
print(i)

#Accesos a los elementos de un array
#Array unidimensional
array_uni= np.array([1,3,5,7,9,11])
print("Shape: ",array_uni.shape)
print("array_uni: ",array_uni)
#Accediendo al quinto elemento
print(array_uni[4])

#Accediendo a los elementos 0, 3 y 5 del array
print(array_uni[0::3])

#Array multidimensional
array_multi= np.array([[1,2,3,4],[5,6,7,8]])
print("Shape:", array_multi.shape)
print("Array multi:\n", array_multi)

#Accediendo al cuarto elemento del array multi
print(array_multi[0,3])

#Accediendo a los datos de la segunda fila
print(array_multi[1,:])

#Accediendo al tercer elemento de las dos primeras filas del array
print(array_multi[0:2, 2])

#Modificación de un array/matriz
#Crear un array unidimensional inicializando un 
#rango de elementos del 0 al 27

array1=np.arange(24)
print("Shape:",array1.shape)
print("Array 1:\n",array1)
#Cambiar las dimensiones del array
array1.shape=(6,4)
print("Shape:",array1.shape)
print("Array 1:\n",array1)

#Cambiar las dimensiones del array y sus longitudes
array2=array1.reshape(4,6)
print("Shape:",array2.shape)
print("Array 2:\n",array2)


