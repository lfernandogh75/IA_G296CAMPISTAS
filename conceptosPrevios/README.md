# Comandos
```
python --version
Python 3.9.2
git --version
https://git-scm.com/
pip list
python.exe -m pip install --upgrade pip
```
## Crear entorno virtual 
python -m venv env
## Activar en torno
env\scripts\activate
si no activa ejecute en el powershell
set-ExecutionPolicy Unrestricted
y escriba S

## Estructura de datos en Pandas
| Tipo      | Contenido                                     |
| --------- | --------------------------------------------- |
| Series    | Array de una demensión                        |
| DataFrame | se corresponde con una tabla de 2 dimensiones |
| panel     | Similar a un diccionario de DataFrame         |

# Creación de objetos Series
```
import pandas as pd # pip install pandas
# Creación de objeto Serie
s= pd.Series([2,4,6,8,10])
print(s)

```
manejo de git
git init
git add .
git commit -m "introdución a pandas 5%"
si sale error 
  git config --global user.email "you@example.com"
  git config --global user.name "Your Name"


  crear archivo requirements.txt
  pip freeze > requirements.txt
 clonar repositorio
 https://github.com/lfernandogh75/IA_G296CAMPISTAS.git
 como desactivar el entorno virtual
  >deactivate

  Recuperar librerias
  con env ejecutado aplicamos
  pip install -r requirements.txt