# Predicción de Precios Dinámicos de Uber

**Sistema de Precios Dinámicos para Viajes en Uber**

Este proyecto implementa un pipeline completo de Machine Learning para predecir el precio de los viajes de Uber (Booking Value). El proceso incluye la carga de datos, un preprocesamiento avanzado, optimización de hiperparámetros de un modelo GradientBoostingRegressor, y el despliegue del modelo final como una API REST usando FastAPI y ngrok.

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c6c9c44d-59e2-4012-b476-9ee096927806" />

**Stack Tecnológico**

* Análisis y Modelado: Python, Pandas, Numpy, Scikit-learn (para GradientBoostingRegressor, GridSearchCV, OneHotEncoder), Joblib.
* Visualización: Matplotlib, Seaborn.
* Carga de Datos: KaggleHub.
* Despliegue API: FastAPI, Uvicorn.
* Túnel (Hosting): Ngrok (para exponer la API desde Colab).

**Metodología y Enfoque Técnico**
El núcleo del sistema es un modelo de regresión entrenado con el algoritmo 

Gradient Boosting. Este método de ensemble learning construye árboles de decisión de forma secuencial, donde cada nuevo árbol corrige los errores del anterior, resultando en un modelo altamente preciso.

Variables Predictoras Clave:
* Hora del día 
* Día de la semana 
* Tipo de vehículo 
* Distancia del viaje 
* Demanda en la zona 

La precisión del modelo se mide utilizando el Error Cuadrático Medio (MSE).


**Dashboard dinamico que permite tomar decisiones de negocio.**
<img width="1274" height="700" alt="image" src="https://github.com/user-attachments/assets/d205268a-a90e-4c8a-bc08-b2c4f0bb1cfc" />


