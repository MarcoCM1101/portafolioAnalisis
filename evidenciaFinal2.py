# Importar librerías
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns


def cleanData(path):
    '''Función para cargar y limpiar los datos'''
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(path)

    # Convertir la columna 'loan_status' a valores binarios (1 para "Approved", 0 para "Rejected")
    df['loan_status'] = df['loan_status'].replace(
        {' Approved': 1, ' Rejected': 0})

    # Columnas relevantes con más correlación
    X = df[['cibil_score', 'loan_term', 'no_of_dependents']]
    y = df['loan_status']  # Variable objetivo

    # Separar los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model():
    '''Función para entrenar los modelos de clasificación'''

    # Variables globales para los modelos y los datos
    global scaler, log_reg, tree, forest, gradient, X_train, X_test, y_train, y_test

    # Mandar a llamara la función cleanData para poder liompiar los datos
    X_train, X_test, y_train, y_test = cleanData(
        'extended_loan_approval_dataset_20k.csv')

    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar modelos
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)

    gradient = GradientBoostingClassifier()
    gradient.fit(X_train, y_train)

    # Datos de prueba para validar los modelos, con un buen y un mal historial crediticio
    # Buen historial crediticio
    cibil_score_good = 778
    loan_term_good = 12
    dependents_good = 2

    # Escalar los datos
    input_data_good = scaler.transform(
        [[cibil_score_good, loan_term_good, dependents_good]])

    # Mal historial crediticio
    cibil_score_bad = 417
    loan_term_bad = 8
    dependents_bad = 0

    # Escalar los datos
    input_data_bad = scaler.transform(
        [[cibil_score_bad, loan_term_bad, dependents_bad]])

    # Evaluar los modelos
    print("==============================================")
    print("               Evaluación de Modelos          ")
    print("==============================================\n")

    # Logistic Regression
    print("🔹 Logistic Regression:")
    print(f"  - Exactitud: {log_reg.score(X_test, y_test):.2f}")
    print(
        f"  - Matriz de Confusión:\n{confusion_matrix(y_test, log_reg.predict(X_test))}\n")
    print(
        f"  - Reporte de Clasificación:\n{classification_report(y_test, log_reg.predict(X_test))}\n")
    print(
        f" - Validate (Approved): {'Approved'if log_reg.predict(input_data_good) == 1 else 'Rejected'}")
    print(
        f" - Validate (Rejected): {'Rejected'if log_reg.predict(input_data_bad) == 0 else 'Approved'}")

    print("----------------------------------------------\n")

    # Decision Tree
    print("🔹 Decision Tree:")
    print(f"  - Exactitud: {tree.score(X_test, y_test):.2f}")
    print(
        f"  - Matriz de Confusión:\n{confusion_matrix(y_test, tree.predict(X_test))}\n")
    print(
        f"  - Reporte de Clasificación:\n{classification_report(y_test, tree.predict(X_test))}\n")
    print(
        f" - Validate (Approved): {'Approved' if tree.predict(input_data_good) == 1 else 'Rejected'}")
    print(
        f" - Validate (Rejected): {'Rejected' if tree.predict(input_data_bad) == 0 else 'Approved'}")
    print("----------------------------------------------\n")

    # Random Forest
    print("🔹 Random Forest:")
    print(f"  - Exactitud: {forest.score(X_test, y_test):.2f}")
    print(
        f"  - Matriz de Confusión:\n{confusion_matrix(y_test, forest.predict(X_test))}\n")
    print(
        f"  - Reporte de Clasificación:\n{classification_report(y_test, forest.predict(X_test))}\n")
    print(
        f" - Validate (Approved): {'Approved' if forest.predict(input_data_good) == 1 else 'Rejected'}")
    print(
        f" - Validate (Rejected): {'Rejected' if forest.predict(input_data_bad) == 0 else 'Approved'}")
    print("----------------------------------------------\n")

    # Gradient Boosting
    print("🔹 Gradient Boosting:")
    print(f"  - Exactitud: {gradient.score(X_test, y_test):.2f}")
    print(
        f"  - Matriz de Confusión:\n{confusion_matrix(y_test, gradient.predict(X_test))}\n")
    print(
        f"  - Reporte de Clasificación:\n{classification_report(y_test, gradient.predict(X_test))}\n")
    print(
        f" - Validate (Approved): {'Approved' if gradient.predict(input_data_good) == 1 else 'Rejected'}")
    print(
        f" - Validate (Rejected): {'Rejected' if gradient.predict(input_data_bad) == 0 else 'Approved'}")
    print("==============================================\n")


def plot_learning_curve(model, model_name):
    '''Función para graficar la curva de aprendizaje'''
    try:
        # Obtener curvas de aprendizaje
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))

        # Calcular promedios y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Graficar la curva de aprendizaje
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Train')
        plt.plot(train_sizes, test_mean, 'o--', label='Test')
        plt.fill_between(train_sizes, train_mean - train_std,
                         train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std,
                         test_mean + test_std, alpha=0.1)
        plt.xlabel("Tamaño del Conjunto de Entrenamiento")
        plt.ylabel("Puntaje")
        plt.title(f"Curva de Aprendizaje - {model_name}")
        plt.legend(loc="best")
        plt.show()

    except Exception as e:
        messagebox.showerror(
            "Error", f"Error al generar curva para {model_name}: {str(e)}")


def predict_with_model(model):
    '''Función para realizar predicciones con un modelo seleccionado'''
    try:
        # Obtener los datos de entrada
        cibil_score = float(entry_cibil.get())
        loan_term = float(entry_loan_term.get())
        no_of_dependents = float(entry_dependents.get())

        # Escalar los datos
        input_data = scaler.transform(
            [[cibil_score, loan_term, no_of_dependents]])

        # Realizar la predicción con el modelo seleccionado
        prediction = model.predict(input_data)

        # Mostrar el resultado
        if prediction == 1:
            messagebox.showinfo("Resultado", "El préstamo fue aprobado")
        else:
            messagebox.showinfo("Resultado", "El préstamo fue rechazado")

    except Exception as e:
        messagebox.showerror("Error", str(e))

        # Nueva función para graficar las matrices de confusión


def plot_confusion_matrix(model, model_name):
    '''Función para mostrar la matriz de confusión en una gráfica'''
    try:
        # Generar predicciones
        y_pred = model.predict(X_test)

        # Crear la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Graficar la matriz de confusión
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))


def exit_app():
    root.quit()  # Cierra la ventana de tkinter


# Crear la ventana de tkinter (Interfaz Gráfica)
root = tk.Tk()
root.title("Préstamo Aprobado/Rechazado")
root.geometry("500x700")

# Mandar a llamar la función train_model para entrenar los modelos
train_model()

plot_confusion_matrix(log_reg, 'Logistic Regression')
plot_confusion_matrix(tree, 'Decision Tree')
plot_confusion_matrix(forest, 'Random Forest')
plot_confusion_matrix(gradient, 'Gradient Boosting')


# Etiquetas y campos de entrada
label_cibil = tk.Label(root, text="Puntaje Crediticio:")
label_cibil.pack(pady=5)
entry_cibil = tk.Entry(root)
entry_cibil.pack(pady=5)

label_loan_term = tk.Label(root, text="Plazo de Préstamo (años):")
label_loan_term.pack(pady=5)
entry_loan_term = tk.Entry(root)
entry_loan_term.pack(pady=5)

label_dependents = tk.Label(root, text="Cantidad de dependientes:")
label_dependents.pack(pady=5)
entry_dependents = tk.Entry(root)
entry_dependents.pack(pady=5)

# Botón para predecir con Logistic Regression
button_logistic = tk.Button(root, text="Predecir con Logistic Regression",
                            command=lambda: predict_with_model(log_reg))
button_logistic.pack(pady=10)

# Botón para predecir con Decision Tree
button_tree = tk.Button(root, text="Predecir con Decision Tree",
                        command=lambda: predict_with_model(tree))
button_tree.pack(pady=10)

# Botón para predecir con Random Forest
button_forest = tk.Button(root, text="Predecir con Random Forest",
                          command=lambda: predict_with_model(forest))
button_forest.pack(pady=10)

# Botón para predecir con Gradient Boosting
button_gradient = tk.Button(root, text="Predecir con Gradient Boosting",
                            command=lambda: predict_with_model(gradient))
button_gradient.pack(pady=10)

# Botones para mostrar la curva de aprendizaje para cada modelo
button_logistic_curve = tk.Button(root, text="Curva de Aprendizaje - Logistic Regression",
                                  command=lambda: plot_learning_curve(log_reg, 'Logistic Regression'))
button_logistic_curve.pack(pady=10)

button_tree_curve = tk.Button(root, text="Curva de Aprendizaje - Decision Tree",
                              command=lambda: plot_learning_curve(tree, 'Decision Tree'))
button_tree_curve.pack(pady=10)

button_forest_curve = tk.Button(root, text="Curva de Aprendizaje - Random Forest",
                                command=lambda: plot_learning_curve(forest, 'Random Forest'))
button_forest_curve.pack(pady=10)

button_gradient_curve = tk.Button(root, text="Curva de Aprendizaje - Gradient Boosting",
                                  command=lambda: plot_learning_curve(gradient, 'Gradient Boosting'))
button_gradient_curve.pack(pady=10)

# Botón para salir
button_exit = tk.Button(root, text="Salir", command=exit_app)
button_exit.pack(pady=10)

# Iniciar el loop de tkinter
root.mainloop()
