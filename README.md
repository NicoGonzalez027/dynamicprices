# Sistema de Precios Dinámicos de Uber


Proyecto que implementa un pipeline completo de Machine Learning para predecir el precio de viajes de Uber, entrenar y optimizar un modelo de `GradientBoostingRegressor` y desplegarlo como API REST con **FastAPI**. Pensado para ejecutarse en Google Colab y exponer la API con **ngrok**.

---

## Contenido de este README

1. Visión general
2. Estructura del repositorio
3. Requerimientos
4. Carga y Limpieza de Datos
5. Ingeniería de Características
6. Preprocesamiento Avanzado
7. Optimización del Modelo
8. Evaluación y Resultados
9. Persistencia del Modelo
10. Despliegue de la API con FastAPI
11. Cómo Ejecutar el Proyecto en Google Colab
12. Persistencia y Consejos
13. Resultados
---

## 1. Visión general

Este repositorio contiene todo lo necesario para reproducir el entrenamiento y despliegue de un modelo de precios dinámicos para viajes. Está pensado para:

* Reproducir preprocesamiento y feature engineering.
* Ejecutar búsqueda de hiperparámetros (GridSearchCV).
* Serializar modelo y encoder con `joblib`.
* Exponer el modelo en una API REST (`FastAPI`).
* Ejecutarlo desde Google Colab y publicar una URL pública con `ngrok`.

## 2. Estructura del repositorio

```
uber-dynamic-pricing/
├─ data/                       # Datos crudos y procesados
├─ notebooks/                  # Notebooks exploratorios y entrenamiento
│  ├─ Proyecto_uber.ipynb
├─ .gitignore
├─ README.md
```

---

## 3. Librerías usadas

```text
numpy
pandas
scikit-learn
joblib
fastapi
uvicorn
python-multipart
pydantic
matplotlib
seaborn
ngrok
pytest
black
flake8
```

---

## 4. Carga y Limpieza de Datos

* Se utiliza **kagglehub** para descargar el dataset `ncr_ride_bookings.csv`.
* Se filtran los datos para incluir únicamente los viajes con estado `Completed`.
* Se combinan las columnas **Date** y **Time** en un único objeto **DateTime** para facilitar la extracción de características temporales.

---

## 5. Ingeniería de Características (Feature Engineering)

Para mejorar la capacidad predictiva del modelo, se crearon las siguientes características:

### **Características Temporales:**

* **Hour:** Hora del día (0-23)
* **DayOfWeek:** Día de la semana (0-6)
* **Month:** Mes del año
* **IsWeekend:** 1 si es sábado o domingo, 0 en caso contrario
* **IsPeakHour:** 1 si está entre las horas pico [7–9, 17–19]
* **IsHolidayHour:** 1 si es horario nocturno o valle [0–5, 23]
* **IsBusinessHour:** 1 si es horario laboral estándar [8–17]

### **Características de Interacción:**

* **Distance_Peak:** `Ride Distance * IsPeakHour`
* **Distance_Weekend:** `Ride Distance * IsWeekend`

### **Codificación Categórica:**

* **Vehicle Type** se codifica usando `OneHotEncoder` para convertir los tipos de vehículo en columnas numéricas.

---

## 6. Preprocesamiento Avanzado

* **Manejo de Outliers:** Se realizó un análisis de la variable objetivo (**Booking Value**) y se decidió filtrar los valores atípicos extremos. Se conservaron únicamente los datos entre el **percentil 5 y el 95** para estabilizar el modelo.
* **Transformación Logarítmica:** Dado que la variable objetivo presentaba una distribución sesgada, se aplicó una transformación `np.log1p` para normalizarla. Esto mejora el aprendizaje del modelo y reduce el error.

---

## 7. Optimización del Modelo

* **Modelo Base:** `GradientBoostingRegressor`.
* **Optimización de Hiperparámetros:** Se implementó `GridSearchCV` para encontrar la mejor combinación de parámetros (`n_estimators`, `learning_rate`, `max_depth`, etc.) que minimizara el **Error Cuadrático Medio (MSE)**.

---

## 8. Evaluación y Resultados

El modelo fue entrenado con las características optimizadas y la transformación logarítmica. Para la evaluación, las predicciones se revirtieron a su escala original con `np.expm1`.

El resultado fue una **reducción significativa del MSE** en comparación con un modelo base, demostrando la efectividad del **feature engineering** y la optimización.

### **Importancia de Características**

Las características más influyentes fueron:

* `Ride Distance`
* `Distance_Weekend`
* `Hour`
* `Distance_Peak`
* `DayOfWeek`

---

## 9. Persistencia del Modelo

El modelo optimizado (`best_model`) y el codificador (`encoder`) se guardaron como archivos `.pkl` usando `joblib`, para su posterior uso en la API.

---

## 10. Despliegue de la API con FastAPI

El proyecto incluye una **API REST** para consumir el modelo entrenado.

### **Endpoints de la API**

La API se ejecuta en Google Colab y se expone mediante **ngrok**.

### `GET /`

Endpoint raíz que da la bienvenida a la API.

### `GET /vehicle_types`

Devuelve una lista de los tipos de vehículos que el modelo reconoce (obtenidos del encoder).

**Ejemplo de Respuesta:**

```json
{
  "tipos_vehiculo_disponibles": ["Auto", "Bike", "Mini", "Prime", "Sedan"],
  "total_tipos": 5
}
```

### **Dashboard dinámico para decisiones de negocio**

<img width="1274" height="700" alt="dashboard" src="https://github.com/user-attachments/assets/d205268a-a90e-4c8a-bc08-b2c4f0bb1cfc" />

### `GET /features`

Devuelve la lista completa de todas las características que el modelo espera recibir.

### `POST /predict`

El endpoint principal para realizar predicciones.

**Ejemplo de Request:**

```json
{
  "hour": 20,
  "day_of_week": 5,
  "month": 11,
  "is_weekend": 1,
  "is_peak_hour": 0,
  "is_holiday_hour": 0,
  "is_business_hour": 0,
  "ride_distance": 8.5,
  "distance_peak": 0,
  "distance_weekend": 8.5,
  "vehicle_type": "Auto"
}
```

**Ejemplo de Respuesta:**

```json
{
  "estado": "éxito",
  "precio_predicho": 145.72,
  "moneda": "INR",
  "caracteristicas_utilizadas": 18,
  "tipo_vehiculo": "Auto"
}
```

---

## 11. Cómo Ejecutar el Proyecto en Google Colab

Este proyecto está diseñado para ejecutarse directamente en **Google Colab**.

1. Abre el archivo `.ipynb` en Google Colab.
2. Reemplaza tu token de ngrok (consíguelo gratis en [ngrok.com](https://ngrok.com)).

```python
# Reemplaza con tu token personal de ngrok
ngrok.set_auth_token("TU_NUEVO_TOKEN_AQUÍ")
```

3. Ejecuta todas las celdas en orden.

El script:

* Instalará las librerías necesarias.
* Descargará los datos desde KaggleHub.
* Entrenará y optimizará el modelo (GridSearchCV puede tardar varios minutos).
* Lanzará la API FastAPI y expondrá la URL pública de ngrok.

**Salida esperada:**

```
Public URL: https://<tu-url-ngrok>.ngrok.io
```

Puedes usar esta URL para probar la API desde cualquier cliente (Postman, curl, etc.).

---

## 12. Persistencia y Consejos

* Los archivos `.pkl` generados se guardan en la sesión temporal de Colab.
* Descárgalos manualmente si deseas conservarlos tras reiniciar el entorno.
* No publiques tu token de ngrok en GitHub.

---

## 13. Resultados

El modelo logra predecir precios con alta precisión, aprovechando variables contextuales como hora, día, distancia y tipo de vehículo, permitiendo un sistema de **pricing dinámico** que puede ser usado como base para dashboards o decisiones de negocio.

---

