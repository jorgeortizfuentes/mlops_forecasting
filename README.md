# 🤖 Awto ML Engineering Challenge

## Setup y Envío

- **_Setup_**:
  - Cree un repositorio y sientase libre de commitear su trabajo con la frecuencia que le parezca
  - Estara usando las siguientes herramientas para resolverlo
  - Python 3.8 (& pip)
  - Docker
  - Cualquier otra libreria que decidas usar


- **_Envio_**
  - Puede conpartirnos la url del repositorio con su solucion
  - Puede comprimir su solucion y enviarnosla por mail

Buena Suerte!

## Challenge

### Part 1: Entrenamiento

- En esta parte del challenge usted va a escribir codigo, el objetivo es entrenar y preparar un modelo de machine learning que permita predecir el la enegia producida por una turbina eólica, dado ciertos parametros.

- _Dataset_: [Wind Turbine Data](./data/wind_power_generation.csv). Este es un conjunto de  históricos de una turbina eólica. Contiene varias características meteorológicas, de turbinas y de los rotores. Los datos se registraron desde enero de 2018 hasta marzo de 2020. Las lecturas se registraron en un intervalo de 10 minutos.

- _Task_:
    - [ ] Cree y entrene un modelo de aprendizaje automático para predecir la `potencia activa` (y podemos esperar que se proporcionen otras columnas como características de entrada).
    - [ ] Operationalise training
    
- Sientase libre de usar cualquiera de estas librerias (e.g. `sklearn`, `tensorflow`, `pytorch`, etc...) y cualquiera de los modelos que ofrecen.

### Part 2: API

- En esta parte, usted va a estar disponibilizando el modelo a traves de una API.

- _Tasks_:
  - [ ] Construya una API REST que disponibilice su modelo
    - Elija su mejor modelo de la anterior parte
    - Use herramientas como `flask`, `django`, `fastapi`, etc.
  - [ ] Operationalise serving
    
## Otros
- Esperamos un nivel de codigo que pueda ser depslegado. Nos fijamos en cosas como, estructura del repositorio, buenas practicas, tests, depurabilidad , extensibilidad.
- Por favor automatice y cree contenedores para algunas o todas las partes de su solucion.  
- **Y por favor**, si le surge alguna duda, no dude en consultarnos.


## Referencias

- [Wind Power Forcasting Dataset from Kaggle](https://www.kaggle.com/theforcecoder/wind-power-forecasting)