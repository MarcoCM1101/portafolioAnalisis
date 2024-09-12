# Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.

### A01753729 Marco Antonio Caudillo Morales

Este proyecto es una **aplicación gráfica de predicción de aprobación de préstamos** desarrollada en **Python** utilizando bibliotecas como `tkinter` para la interfaz gráfica de usuario y `scikit-learn` para la implementación de modelos machine learning.

## Descripción

El sistema predice si un préstamo será **aprobado** o **rechazado** basado en el puntaje crediticio del solicitante, la duración del préstamo, y la cantidad de dependientes. Utiliza cuatro modelos de clasificación diferentes:

1. **Regresión Logística**
2. **Árbol de Decisión**
3. **Random Forest**
4. **Gradient Boosting**

Los modelos se entrenan con un conjunto de datos proporcionado y, posteriormente, pueden realizar predicciones en función de los datos de entrada proporcionados por el usuario.

**El modelo que se ha escogido es `Gradient Boosting` puesto que es el modelo con más exactitud y cuya gráfica de curva de aprendizaje fue la que mejores resultados presentaba**

### Funcionalidades:

- **Entrenamiento del modelo**: Entrena cuatro modelos de clasificación diferentes.
- **Predicción**: Realiza predicciones de aprobación o rechazo de préstamos basadas en las entradas del usuario.
- **Curva de aprendizaje**: Muestra las curvas de aprendizaje para cada modelo.
- **Validación de datos predefinidos**: Valida casos de buen y mal historial crediticio en cada modelo.

## Requisitos

Este proyecto requiere las siguientes dependencias:

- **Python 3.x**
- **tkinter**: Biblioteca integrada para crear interfaces gráficas en Python.
- **pandas**: Para la manipulación de datos.
- **numpy**: Para el manejo de arreglos numéricos.
- **matplotlib**: Para las gráficas de las curvas de aprendizaje.
- **seaborn**: Biblioteca para visualización de datos.
- **scikit-learn**: Para entrenar y evaluar los modelos de aprendizaje automático.

Puedes instalar todas las dependencias usando `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Instrucciones para ejecutar

1. **Clonar el repositorio**:
   Clona este proyecto en tu máquina local utilizando el siguiente comando:

   ```bash
   git clone https://github.com/MarcoCM1101/evidencia2.git
   ```

2. **Agregar el archivo CSV**:
   Coloca tu archivo de datos CSV (`extended_loan_approval_dataset_20k.csv`) en la misma carpeta donde está el código. El archivo CSV debe contener las siguientes columnas:

   - **cibil_score**: Puntaje crediticio del solicitante.
   - **loan_term**: Duración del préstamo en meses.
   - **no_of_dependents**: Número de dependientes del solicitante.
   - **loan_status**: Estado del préstamo ('Approved' o 'Rejected').

3. **Ejecutar el script**:
   Ejecuta el archivo Python desde la línea de comandos o cualquier entorno de desarrollo como PyCharm o VSCode:

   ```bash
   python evidenciaFinal2.py
   ```

4. **Interfaz de Usuario**:
   - Ingresa los valores requeridos para **puntaje crediticio**, **duración del préstamo** y **número de dependientes**.
   - Selecciona el modelo deseado para realizar la predicción: Regresión Logística, Árbol de Decisión, Random Forest o Gradient Boosting.
   - Haz clic en los botones correspondientes para predecir o visualizar la curva de aprendizaje de cada modelo.
   - El sistema mostrará si el préstamo es **aprobado** o **rechazado**.

## Estructura del Proyecto

```bash
EVIDENCIA2/
├── Evaluation                  #Folder con screenshots de la evaluación de los modelos
├    ├── Decision_Tree.png
├    ├── Grtadient_Boosting.png
├    ├── Logistic_Regresion.png
├    ├── Random_Forest.png
├── Graphics                  #Folder con screenshots de la gráficas de los modelos
├    ├── Decision_Tree.png
├    ├── Grtadient_Boosting.png
├    ├── Logistic_Regresion.png
├    ├── Random_Forest.png
├── evidenciaFinal2.py          # Archivo principal del sistema de predicción.
├── extended_loan_approval_dataset_20k.csv  # Dataset de ejemplo para el entrenamiento.
├── LICENSE                 # Archivo de Licencia.
├── README.md                 # Archivo de documentación.
```

## Funciones Principales

1. **cleanData(path)**: Carga y limpia los datos del archivo CSV, y los divide en conjuntos de entrenamiento y prueba.
2. **train_model()**: Entrena los modelos de clasificación, calcula métricas de evaluación y realiza validaciones con datos de ejemplo.

3. **plot_learning_curve(model, model_name)**: Genera las curvas de aprendizaje para cada modelo y las muestra utilizando `matplotlib`.

4. **predict_with_model(model)**: Realiza predicciones basadas en los datos proporcionados por el usuario a través de la interfaz gráfica.

5. **Interfaz gráfica (`tkinter`)**: Permite al usuario ingresar datos y seleccionar el modelo con el que desea realizar predicciones.

## Modelos Implementados

1. **Logistic Regression**: Un modelo lineal para clasificación binaria.
2. **Decision Tree**: Modelo no lineal basado en árboles de decisión.
3. **Random Forest**: Un conjunto de múltiples árboles de decisión para mejorar la precisión.
4. **Gradient Boosting**: Algoritmo de ensamblaje que combina varios modelos débiles para formar un modelo fuerte.

## Ejemplo de Validación

El sistema viene preconfigurado para validar los modelos con ejemplos de buen y mal historial crediticio:

- **Buen historial crediticio**: `cibil_score=778`, `loan_term=12`, `dependents=2`
- **Mal historial crediticio**: `cibil_score=417`, `loan_term=8`, `dependents=0`

Cada modelo evaluará estos ejemplos para mostrar si el préstamo es aprobado o rechazado.

## Imagen de la interfaz Gráfica

![Imagen de Interfaz Gráfica](<Screenshot 2024-09-09 at 22.24.48.png>)
