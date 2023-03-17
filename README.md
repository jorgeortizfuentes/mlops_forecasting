# AWTO MLE Challenge

Autor: Jorge Luis Ortiz-Fuentes

## Descripción

Este es un desafío de aprendizaje automático de dos partes. Primero, se entrenaron modelos para predecir la energía producida por un aerogenerador dadas ciertas variables. En la segunda parte, se construyó una API para que el modelo esté disponible en línea.

## Hardware y versiones

Este código se diseñó y se testeó usando Python 3.8.16 en una máquina con Ubuntu 22.04 con 2 GPU NVIDIA RTX A6000.

## Parte 1: Entrenamiento

En esta parte del desafío se debe escribir código para entrenar y preparar un modelo de aprendizaje automático que permita predecir la energía producida por un aerogenerador dadas ciertas variables.

### Datos

El conjunto de datos utilizado en este desafío se encuentra en `./data/wind_power_generation.csv`. Este conjunto de datos contiene varias variables meteorológicas, de rotor y de turbina. Los datos se registraron desde enero de 2018 hasta marzo de 2020 en intervalos de 10 minutos.

### Tarea

- Crear y entrenar un modelo de aprendizaje automático para predecir la `potencia activa` (y se pueden proporcionar otras columnas como variables de entrada).
- Operacionalizar el entrenamiento.

## Experimentos

En paralelo al desarrollo del código, se crearon `Jupyter Notebooks` para explorar el dataset, probar clasificadores y hacer inferencias.

## Organización del código y patrones de diseño

La organización del código sigue una estructura de paquetes en la que se separan los módulos de entrenamiento, prediccion, preprocesamiento y búsqueda de hiperparámetros.

Se ha utilizado el patrón "Single Responsibility Principle" para separar las responsabilidades de cada módulo.

El módulo `preprocess.py` se encarga de preprocesar los datos en bruto obtenidos en el archivo `wind_power_generation.csv`.

El módulo `search_hyperparameters.py` se encarga de realizar una búsqueda de hiperparámetros utilizando la biblioteca Optuna.

El módulo `trainer.py` se encarga de entrenar el modelo con los mejores hiperparámetros encontrados en la búsqueda de hiperparámetros y guardar el modelo entrenado.

El módulo `predict.py` se encarga de cargar el modelo entrenado y realizar predicciones.

Se ha utilizado la biblioteca Darts para implementar el modelo de predicción y PyTorch Lightning para su entrenamiento.

Se ha utilizado el patrón "Inyección de Dependencias" para inyectar las dependencias necesarias en cada módulo y hacerlos más independientes y fáciles de testear.

Se ha utilizado la biblioteca Optuna para realizar una búsqueda de hiperparámetros y la biblioteca Plotly para visualizar los resultados de la búsqueda.

### Preprocesamiento de los datos

El preprocesamiento de datos es una etapa fundamental para realizar predicciones, ya que permite asegurarse de que los datos estén limpios, completos y estructurados de manera adecuada para poder ser utilizados en la tareas de modelado. En este caso, se han utilizado los siguientes métodos de preprocesamiento:

#### Convertir fechas

Se ha convertido la columna "ds" a formato de fecha y se ha eliminado la zona horaria.

### Rellenar datos faltantes

Se han rellenado los valores faltantes en las otras columnas con el valor 0. Se podrían haber probado otras formas, tales como medidas de estadística descriptiva para reemplazar los valores. Sin embargo, esto se deja como una tarea pendiente para el futuro.

### Eliminar columnas duplicadas y las irrelevantes

Se han eliminado las columnas "Blade3PitchAngle" y "WindDirection", ya que contienen información duplicada. Se eliminó también la columna "WTG" ya que no contiene información útil.

## Clasificadores

Para este desafío se probaron 5 técnicas para realizar predicciones multivariadas:

- TCN (Temporal Convolutional Network): es una red neuronal convolucional que se utiliza para analizar series de tiempo y realizar predicciones.

- XGBModel (XGBoost): es un modelo de aprendizaje de conjunto que utiliza árboles de decisión para realizar predicciones.

- RNNModel (Recurrent Neural Network): es una red neuronal que se utiliza para analizar secuencias y series de tiempo, y realizar predicciones.

- LSTM (Long Short-Term Memory): es una variante de las redes neuronales recurrentes que se utiliza para analizar secuencias y series de tiempo, y realizar predicciones.

- GRU (Gated Recurrent Unit): es otra variante de las redes neuronales recurrentes que se utiliza para analizar secuencias y series de tiempo, y realizar predicciones.

Los resultados se presentan en la siguiente tabla.

| model    | mae        | rmse       |
| -------- | ---------- | ---------- |
| TCN      | 347.733438 | 505.100243 |
| XGBModel | 668.633280 | 774.615943 |
| RNNModel | 403.156471 | 602.109330 |
| LSTM     | 403.189509 | 602.155056 |
| GRU      | 415.720545 | 615.308414 |

## Métricas

Las métricas utilizadas para evaluar la precisión de los modelos fueron MAE (Mean Absolute Error) y RMSE (Root Mean Square Error).

La MAE se calcula mediante la suma de las diferencias absolutas entre las predicciones del modelo y los valores reales, dividida por el número total de observaciones. Es decir, mide la magnitud promedio de los errores en un conjunto de predicciones, sin considerar su dirección. Un valor de MAE más bajo indica una mayor precisión del modelo.

El RMSE es similar a la MAE, pero tiene en cuenta la magnitud de los errores al elevarlos al cuadrado antes de calcular su promedio y luego tomar su raíz cuadrada. Esto significa que el RMSE penaliza más los errores grandes y, por lo tanto, es más sensible a las desviaciones extremas. Al igual que la MAE, un valor de RMSE más bajo indica una mayor precisión del modelo.

## Optimización de hiperparámetros

A partir de los experimentos realizados en los notebooks se observó que el modelo que entregó mejores resultados fue TCN. Este modelo utiliza capas convolucionales 1D para procesar secuencias de datos en una ventana de tiempo, lo que permite capturar patrones a diferentes escalas de tiempo en una serie de tiempo. Además, el modelo utiliza una técnica conocida como dilatación causal, que permite aumentar el tamaño efectivo de la ventana de tiempo que se procesa mientras se mantiene el mismo tamaño de ventana de tiempo en las capas convolucionales.

Para entrenar el modelo final, se utilizó una búsqueda de hiperparámetros con la biblioteca Optuna, que utiliza técnicas bayesianas para optimizar la búsqueda de las mejores métricas. Esto permitió encontrar la combinación óptima de hiperparámetros para minimizar el error de predicción RMSE en un conjunto de validación. Los hiperparámetros que se optimizaron fueron: la longitud de la ventana de entrada, el número de capas convolucionales, el número de filtros en cada capa, el dropout y el número de épocas de entrenamiento.

Todas las combinaciones de hiperparámetros y las métricas obtenidas se guardaron en el archivo `hyperparameters_results.csv`. Estos valores son utilizados posteriormente para el entrenamiento del modelo final.

El mejor modelo obtuvo de resultado 461.65 de `RMSE`.

## Entrenamiento final

El archivo `train.py` contiene la clase Trainer, que se encarga de cargar, preprocesar y entrenar los datos utilizando un modelo TCN (Temporal Convolutional Network) de la librería Darts. También se encarga de guardar el modelo entrenado y su información en una carpeta especificada en el archivo `config.py`.

A partir de los resultados obtenidos en la búsqueda de hiperparámetros, la Clase Trainer entrena unn modelo con el 100% de los datos y lo guarda en la carpeta `models` junto a la información del modelo y sus hiperparámetros en un archivo `models/models_info.csv`.

### Inferencias

El código para hacer inferencias se encuentra en el archivo `predict.py`. Este archivo contiene la clase `PowerPredictor`, que es una clase para hacer predicciones de la salida de energía de un modelo de aprendizaje automático entrenado previamente.

La clase `PowerPredictor` tiene dos métodos:

1. `get_time_position`: Este método se encarga de calcular la cantidad de predicciones que se deben hacer, basándose en la fecha de la última predicción del modelo y la fecha de la predicción que se quiere hacer. Retorna el número de predicciones a hacer.

32 `predict_power_output`: Este método se encarga de hacer las predicciones de la salida de energía. Recibe como parámetro la fecha para la cual se quiere hacer la predicción. Primero, se llama al método `get_time_position` para saber cuántas predicciones hacer. Luego, se hace la predicción llamando al método `predict` del modelo cargado en el objeto `PowerPredictor`. Finalmente, se convierte el resultado a un diccionario y se retorna.

El código incluye un ejemplo de uso al final del archivo. En este ejemplo, se crea un objeto `PowerPredictor` y se hace una predicción para una fecha determinada. El resultado se imprime en la consola.

### Limitación de las inferencias

El modelo solo logra predecir 887 predicciones en intervalos cada 10 minutos. Por lo tanto, el modelo solo permite realizar inferencias entre 2020-03-30 23:50:00 (última fecha del dataset de entrenamiento) y 2020-04-06 03:40.
## Parte 2: API

En esta parte, se construyó una API REST con FastAPI para disponibilizar el modelo en línea. Se prefirió este framework en comparación a Django y Flask por su mayor rendimiento, facilidad de uso y su posibilidad de escalabilidad.

## Uso

La API de predicción de energía eólica funciona de la siguiente manera:

1. El usuario envía una solicitud GET a la ruta `/predict/{date}` donde `{date}` es la fecha y hora para la cual se desea hacer la predicción en formato "YYYY-MM-DD HH:MM".
2. La API verifica que la fecha y hora estén en el formato correcto y que los minutos sean múltiplos de 10. Si no es así, redondea los minutos al múltiplo de 10 superior.
3. La API utiliza el modelo TCN entrenado previamente y el objeto `PowerPredictor` para realizar la predicción de la energía eólica para la fecha y hora especificada.
4. La API devuelve la predicción en formato JSON con la siguiente estructura:

```json
[
  { "ds": "2020-03-31 00:00", "ActivePower": -366.71939457844417 },
  { "ds": "2020-03-31 00:10", "ActivePower": -144.11230095788068 },
  { "ds": "2020-03-31 00:20", "ActivePower": 625.2472260190992 },
  { "ds": "2020-03-31 00:30", "ActivePower": 1735.9998637433953 },
  { "ds": "2020-03-31 00:40", "ActivePower": 2983.776382668153 }
]
```

### Tests

Se crearon tests para probar los scripts `train/preprocess.py`, `train/trainer.py`, `train/predict.py` y `api/main.py`. Se pueden correr todos los tests con el siguiente comando en consola:

```bash
python -m unittest discover tests/
```

### Docker

La API se puede inicializar de dos maneras:

1. Ejecutar el archivo `main.py`para probar la API. No se recomienda este método para poner la API en producción.

2. Ejecutar la API a través de Docker con `uvicorn`. Para ello se debe crear el container y luego ejecutar.

```bash
docker build -t energy-api .
```

Correr el container:

```
docker run -p 8282:8282 energy-api
```

Realiza una petición GET a la siguiente URL: "http://localhost:8282/predict/{fecha y hora}" (sustituye "{fecha y hora}" por la fecha y hora en formato "YYYY-MM-DD HH:MM"). La API devolverá un JSON con la predicción de la producción de energía correspondiente a la fecha y hora especificadas.

Por ejemplo, si quieres predecir la producción de energía para el 31 de marzo de 2020 a las 04:50, deberás hacer una petición GET de la siguiente forma "2020-03-31 04:50".

Disclaimer: considerar la limitación presentada en la sección de Inferencias.

## Posibles mejoras

A continuación, se presentan algunas ideas de trabajo futuro para seguir mejorando el modelo y el deployment:

- Experimentar con diferentes técnicas de preprocesamiento de datos y ver cómo afectan el rendimiento del modelo.

- Probar con diferentes hiperparámetros en los modelos en los modelos de RNN, LSTM y GRU.

- Utilizar una arquitectura de microservicios para desplegar la API en varios servidores y mejorar su escalabilidad.

- Utilizar técnicas de optimización de modelos en línea para actualizar el modelo de manera continua a medida que se obtienen nuevos datos.

- Utilizar caching para almacenar en caché los resultados de las consultas a la API. Esto puede mejorar significativamente el rendimiento de la API al reducir el número de consultas que deben realizarse al modelo para las mismas entradas.

- Implementar una base de datos para guardar las predicciones y los datos de entrenamiento en lugar de cargar los datos directamente en la memoria.

- Entrenar el modelo con un `output_chunk_length` superior para permitirle al modelo predecir un periodo más extenso en el futuro.
