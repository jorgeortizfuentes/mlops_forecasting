# AWTO MLE Challenge

## Descripción

Este es un desafío de aprendizaje automático de dos partes. Primero, se entrenaron modelos para predecir la energía producida por un aerogenerador dadas ciertas variables. En la segunda parte, se construyó una API para que el modelo esté disponible en línea.

# Hardware y versiones

Este código se diseñó y se testeó usando Python 3.8.16 en una máquina con Ubuntu 22.04 con 2 GPU NVIDIA RTX A6000.

## Parte 1: Entrenamiento

En esta parte del desafío se debe escribir código para entrenar y preparar un modelo de aprendizaje automático que permita predecir la energía producida por un aerogenerador dadas ciertas variables.

### Datos

El conjunto de datos utilizado en este desafío se encuentra en `./data/wind_power_generation.csv`. Este conjunto de datos contiene varias variables meteorológicas, de rotor y de turbina. Los datos se registraron desde enero de 2018 hasta marzo de 2020 en intervalos de 10 minutos.

### Tarea

- Crear y entrenar un modelo de aprendizaje automático para predecir la `potencia activa` (y se pueden proporcionar otras columnas como variables de entrada).
- Operacionalizar el entrenamiento.

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

## Parte 2: API

En esta parte, se construyó una API REST para disponibilizar el modelo en línea.
