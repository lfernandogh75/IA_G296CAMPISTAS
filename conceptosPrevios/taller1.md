# 🧪 Taller Práctico: Análisis de Datos con Pandas
Nivel: Básico - Intermedio
Objetivo: Aplicar los conceptos de Series y DataFrame en Pandas utilizando una base de datos ficticia de estudiantes.

## 📁 Dataset: Información de Estudiantes
 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset ficticio
datos_estudiantes = {
    "peso": pd.Series([55, 68, 74, 60, 72], index=["Ana", "Carlos", "Daniela", "Eduardo", "Fernanda"]),
    "altura": pd.Series([162, 175, 168, 180, 170], index=["Ana", "Carlos", "Daniela", "Eduardo", "Fernanda"]),
    "promedio": pd.Series([4.5, 3.8, 4.2, 2.9, 3.5], index=["Ana", "Carlos", "Daniela", "Eduardo", "Fernanda"]),
    "edad": pd.Series([17, 18, 17, 19, 18], index=["Ana", "Carlos", "Daniela", "Eduardo", "Fernanda"])
}

df = pd.DataFrame(datos_estudiantes)
print(df)
```
## 🧩 Actividades
1. Crear una Serie con los nombres y alturas de los estudiantes
```
 
```
Pregunta: ¿Cuál es la altura de Daniela?

2. Accede al promedio de calificación de Carlos de 3 formas diferentes:
```
 
```
3. Filtra a los estudiantes con promedio mayor o igual a 4.0
 ```
 
```
Pregunta: ¿Cuántos estudiantes tienen un buen promedio?

4. Calcula operaciones estadísticas:
```
 
```
5. Agrega una nueva columna que indique si el estudiante es mayor de edad
 ```
 
```
6. Agrega una columna con el año de nacimiento (suponiendo que estamos en 2025)
 ```
 
```
7. Visualiza los promedios de los estudiantes en un gráfico
 ```
df["promedio"].plot(kind="bar", title="Promedio de estudiantes")
plt.xlabel("Estudiante")
plt.ylabel("Nota promedio")
plt.show()
```
8. Filtra a los estudiantes con altura entre 165 y 175 cm
 ```
 
```
9. Copia el DataFrame y elimina la columna "peso"
 ```
 
```
10. Crea un nuevo DataFrame con solo 3 columnas: nombre, edad y año de nacimiento
 ```
 
```