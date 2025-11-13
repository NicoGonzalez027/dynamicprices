# Sistema de Precios Dinámicos de Uber

> **Predicción de Booking Value (Precio de Viaje)**

Proyecto que implementa un pipeline completo de Machine Learning para predecir el precio de viajes de Uber, entrenar y optimizar un modelo de `GradientBoostingRegressor` y desplegarlo como API REST con **FastAPI**. Pensado para ejecutarse en Google Colab y exponer la API con **ngrok**.

---

## Contenido de este README

1. Visión general
2. Estructura del repositorio
3. Requerimientos
4. Instrucciones rápidas (Quickstart)
5. Descripción del pipeline
6. Ingeniería de características
7. Preprocesamiento y manejo de outliers
8. Entrenamiento y optimización
9. Evaluación y métricas
10. Endpoints de la API
11. Ejemplos de uso (curl / Python)
12. Despliegue en Colab + ngrok
13. Persistencia y seguridad de credenciales
14. CI / Tests recomendados
15. Contribuir
16. Licencia

---

## 1. Visión general

Este repositorio contiene todo lo necesario para reproducir el entrenamiento y despliegue de un modelo de precios dinámicos para viajes. Está pensado para:

* Reproducir preprocesamiento y feature engineering.
* Ejecutar búsqueda de hiperparámetros (GridSearchCV).
* Serializar modelo y encoder con `joblib`.
* Exponer el modelo en una API REST (`FastAPI`).
* Ejecutarlo desde Google Colab y publicar una URL pública con `ngrok`.

## 2. Estructura recomendada del repositorio

```
uber-dynamic-pricing/
├─ data/                       # (NO subir datos sensibles) csvs, muestras
│  ├─ raw/                     # dataset original (no subir si es privado)
│  └─ processed/               # datos filtrados y listos para modelar
├─ notebooks/                  # Notebooks exploratorios y entrenamiento
│  ├─ 01_exploration.ipynb
│  └─ 02_training_gridsearch.ipynb
├─ src/                        # Código fuente ejecutable
│  ├─ data_loader.py           # funciones para cargar y limpiar
│  ├─ features.py              # feature engineering
│  ├─ preprocessing.py         # pipelines, encoders, transforms
│  ├─ train.py                 # script para entrenar y guardar modelo
│  ├─ predict.py               # wrappers para hacer una predicción local
│  └─ api/                     # código FastAPI
│     ├─ main.py               # app FastAPI
│     ├─ models.py             # pydantic schemas
│     └─ utils.py              # utilidades (carga joblib, etc.)
├─ models/                     # modelos serializados (.pkl)
│  ├─ best_model.pkl
│  └─ encoder.pkl
├─ tests/                      # pruebas unitarias
│  ├─ test_features.py
│  └─ test_api.py
├─ .github/
│  └─ workflows/ci.yml         # GitHub Actions
├─ requirements.txt
├─ .gitignore
├─ README.md
└─ LICENSE
```

---

## 3. Requerimientos

Archivo `requirements.txt` ejemplo (añadir versiones exactas si quieres reproducibilidad):

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

> Recomiendo usar `pip install -r requirements.txt` en un entorno virtual o en Colab.

---

## 4. Instrucciones rápidas (Quickstart)

1. Clona el repositorio:

```bash
git clone https://github.com/tu_usuario/uber-dynamic-pricing.git
cd uber-dynamic-pricing
```

2. Instala dependencias:

```bash
python -m venv venv
source venv/bin/activate   # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

3. Entrena modelo (ejemplo):

```bash
python src/train.py --data_path data/processed/train.csv --output_dir models/
```

4. Ejecuta la API localmente:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

5. (Opcional) Lanza ngrok en Colab y publica la URL.

---

## 5. Descripción del pipeline (resumen técnico)

* **Carga y filtrado**: solo registros con `status == "Completed"`.
* **Datetime**: combinación de `Date` + `Time` a `datetime` y extracción de `hour`, `day_of_week`, `month`.
* **Features temporales**: `is_weekend`, `is_peak_hour`, `is_holiday_hour`, `is_business_hour`.
* **Interacciones**: `distance_peak`, `distance_weekend`.
* **Categorical encoding**: `vehicle_type` -> `OneHotEncoder`.
* **Outliers**: filtrar percentiles 5 y 95 de `booking_value`.
* **Target transform**: `np.log1p(booking_value)` durante entrenamiento y `np.expm1(pred)` al predecir.

---

## 6. Ingeniería de características (detalles)

Incluye funciones reutilizables en `src/features.py`. Ejemplo breve para `is_peak_hour`:

```python
def is_peak(hour: int) -> int:
    return int(hour in list(range(7,10)) + list(range(17,20)))
```

Asegúrate de que el pipeline de features devuelva siempre la misma lista y orden de columnas que el modelo espera. Documenta en `src/api/models.py` el `Pydantic` schema de entrada.

---

## 7. Preprocesamiento y manejo de outliers

* Antes de entrenar, filtrar por percentil 5 y 95 del objetivo.
* Aplicar `np.log1p` a la target para estabilizar varianza.
* Guardar el `scaler/transformer` si lo usas (ej.: `joblib.dump(transformer, "models/transformer.pkl")`).

---

## 8. Entrenamiento y optimización

* Usa `GridSearchCV` con `scoring='neg_mean_squared_error'`.
* Métricas a guardar:

  * Mejor `params_`
  * `best_score_`
  * `cv_results_` (guardar a CSV para análisis)
* Serializar `best_estimator_` y el `encoder`.

---

## 9. Evaluación y métricas

* Entrena/valida con `train_test_split` o `TimeSeriesSplit` si los datos tienen dependencia temporal.
* Reporta:

  * MSE en escala log (train/val)
  * MSE en escala original (tras aplicar `expm1`)
  * MAE, R2
* Genera gráficos: distribución de residuales, importancia de features (SHAP si quieres más profundidad).

---

## 10. Endpoints de la API

### GET /

Bienvenida y versión de la API.

### GET /vehicle_types

Devuelve la lista de `vehicle_type` que conoce el encoder.

**Respuesta ejemplo**:

```json
{
  "tipos_vehiculo_disponibles": ["Auto", "Bike", "Mini", "Prime", "Sedan"],
  "total_tipos": 5
}
```

### GET /features

Lista completa de features que espera el modelo (orden importante).

### POST /predict

Request body (JSON):

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

Respuesta ejemplo:

```json
{
  "estado": "éxito",
  "precio_predicho": 145.72,
  "moneda": "INR",
  "caracteristicas_utilizadas": 18,
  "tipo_vehiculo": "Auto"
}
```

> Nota: valida siempre el schema con Pydantic en `src/api/models.py` para evitar inputs inválidos.

---

## 11. Ejemplos de uso

**curl**:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"hour":20, "day_of_week":5, "month":11, "is_weekend":1, "is_peak_hour":0, "is_holiday_hour":0, "is_business_hour":0, "ride_distance":8.5, "distance_peak":0, "distance_weekend":8.5, "vehicle_type":"Auto"}'
```

**Python (requests)**:

```python
import requests
url = "http://127.0.0.1:8000/predict"
payload = { ... }
r = requests.post(url, json=payload)
print(r.json())
```

---

## 12. Despliegue en Colab + ngrok

* Incluye en el notebook la instalación de `pyngrok` y la llamada `ngrok.set_auth_token("TU_TOKEN")`.
* Evita subir el token a GitHub. Mejor usar variables de entorno o inputs en Colab.
* Ejemplo de comando para exponer la app (en Colab):

```python
!uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)
```

---

## 13. Persistencia y seguridad de credenciales

* **Nunca** subir tokens o credenciales a GitHub.
* Usa `.env` o GitHub Secrets para CI.
* Añade `.env` en `.gitignore`.

Ejemplo `.gitignore` mínimo:

```
venv/
__pycache__/
*.pyc
.env
models/*.pkl
data/raw/
```

---

## 14. CI / Tests recomendados

* Tests unitarios para:

  * `features.py` (valores esperados)
  * `preprocessing.py` (salidas y shapes)
  * `src/api` (endpoints con testclient de FastAPI)
* GitHub Actions: correr `pytest`, `black --check` y `flake8`.

Ejemplo de workflow (`.github/workflows/ci.yml`):

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest -q
```

---

## 15. Contribuir

Incluye un `CONTRIBUTING.md` con:

* Cómo abrir issues y pull requests.
* Estándar de commits.
* Guía de estilo (black / flake8).
* Revisión de PRs.

Breve plantilla de `CONTRIBUTING.md`:

```
1. Fork repo
2. Crear branch feature/<tu-cambio>
3. Mantener tests y documentación actualizada
4. Abrir PR y asignar reviewers
```

---

## 16. Licencia

Añade una licencia (por ejemplo MIT) en `LICENSE`.

---

## Archivos de ejemplo para copiar/pegar

**requirements.txt** (ejemplo corto):

```text
numpy
pandas
scikit-learn
joblib
fastapi
uvicorn
pyngrok
pydantic
pytest
```

**.gitignore** minimal:

```
venv/
__pycache__/
*.pyc
.env
models/*.pkl
data/raw/
```

**src/api/main.py** (esqueleto):

```python
from fastapi import FastAPI
import joblib
from src.api.models import PredictRequest, PredictResponse

app = FastAPI(title="Uber Dynamic Pricing API")

MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/encoder.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Precios Dinámicos"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # transformar input -> vector con mismas columnas esperadas
    # predecir en escala log, aplicar expm1
    return {"estado":"éxito","precio_predicho": 0.0, "moneda":"INR"}
```

---

## Siguientes pasos (sugeridos)

* Añadir notebooks con visualizaciones y resultados de GridSearch.
* Documentar `features.py` con docstrings y tests.
* Añadir un script `make_prediction.py` y ejemplos de requests.
* Integrar SHAP para explicar predicciones si lo deseas.

---


