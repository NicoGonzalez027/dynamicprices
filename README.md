# dynamicprices

**Sistema de Precios Dinámicos para Viajes en Uber**

Este proyecto desarrolla un sistema de precios dinámicos que predice el valor óptimo de un viaje en Uber, basándose en patrones de demanda y características temporales, utilizando técnicas de machine learning.

Tomado a partir de dataset: https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c6c9c44d-59e2-4012-b476-9ee096927806" />


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


