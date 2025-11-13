# Predicción de Precios Dinámicos de Uber

**Sistema de Precios Dinámicos para Viajes en Uber**

Este proyecto implementa un pipeline completo de Machine Learning para predecir el precio de los viajes de Uber (Booking Value). El proceso incluye la carga de datos, un preprocesamiento avanzado, optimización de hiperparámetros de un modelo GradientBoostingRegressor, y el despliegue del modelo final como una API REST usando FastAPI y ngrok.

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c6c9c44d-59e2-4012-b476-9ee096927806" />

# Stack Tecnológico

* Análisis y Modelado: Python, Pandas, Numpy, Scikit-learn (para GradientBoostingRegressor, GridSearchCV, OneHotEncoder), Joblib.
* Visualización: Matplotlib, Seaborn.
* Carga de Datos: KaggleHub.
* Despliegue API: FastAPI, Uvicorn.
* Túnel (Hosting): Ngrok (para exponer la API desde Colab).


# Flujo del Proyecto de Machine Learning
El núcleo de este proyecto es la optimización de un modelo de regresión para reducir el error de predicción (MSE).

1. Carga y Limpieza de Datos
Se utiliza kagglehub para descargar el dataset ncr_ride_bookings.csv.
Se filtran los datos para incluir únicamente los viajes con estado "Completed".

Se combinan las columnas Date y Time en un único objeto DateTime para facilitar la extracción de características temporales.

2. Ingeniería de Características (Feature Engineering)
Para mejorar la capacidad predictiva del modelo, se crearon las siguientes características:

Características Temporales:

Hour: La hora del día (0-23).

DayOfWeek: El día de la semana (0-6).

Month: El mes del año.

IsWeekend: Booleano (1 si es sábado o domingo, 0 si no).

IsPeakHour: Booleano (1 si la hora es considerada "pico" [7-9, 17-19], 0 si no).

IsHolidayHour: Booleano (1 si es horario nocturno/valle [0-5, 23], 0 si no).

IsBusinessHour: Booleano (1 si es horario laboral estándar [8-17], 0 si no).

Características de Interacción:

Distance_Peak: Ride Distance multiplicado por IsPeakHour.

Distance_Weekend: Ride Distance multiplicado por IsWeekend.

Codificación Categórica:

Vehicle Type se codifica usando OneHotEncoder para convertir los tipos de vehículo en columnas numéricas.

3. Preprocesamiento Avanzado
Manejo de Outliers: Se realizó un análisis de la variable objetivo (Booking Value) y se decidió filtrar los valores atípicos extremos. Se conservaron únicamente los datos entre el percentil 5 y el 95 para estabilizar el modelo.

Transformación Logarítmica: La variable objetivo (Booking Value) presentaba una distribución sesgada. Se aplicó una transformación logarítmica (np.log1p) para normalizar su distribución. Esto ayuda al modelo de Gradient Boosting a aprender de manera más efectiva y reducir el error.

4. Optimización del Modelo
Modelo Base: Se utilizó un GradientBoostingRegressor.

Búsqueda de Hiperparámetros: Se implementó GridSearchCV para encontrar la combinación óptima de hiperparámetros (como n_estimators, learning_rate, max_depth, etc.) que minimizara el error cuadrático medio (neg_mean_squared_error).

5. Evaluación y Resultados
El modelo fue entrenado con las características optimizadas y la transformación logarítmica. Para la evaluación, las predicciones se revirtieron a su escala original usando np.expm1.

El resultado fue una reducción significativa del MSE (Error Cuadrático Medio) en comparación con un modelo base no optimizado, demostrando la efectividad del feature engineering y la optimización de hiperparámetros.

Importancia de Características: Las características más influyentes para el modelo optimizado fueron:

Ride Distance

Distance_Weekend

Hour

Distance_Peak

DayOfWeek

6. Persistencia del Modelo
El modelo optimizado (best_model) y el codificador (encoder) se guardaron en archivos .pkl usando joblib para su posterior uso en la API.


**Dashboard dinamico que permite tomar decisiones de negocio.**
<img width="1274" height="700" alt="image" src="https://github.com/user-attachments/assets/d205268a-a90e-4c8a-bc08-b2c4f0bb1cfc" />


