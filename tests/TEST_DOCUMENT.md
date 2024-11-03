# Pruebas Unitarias

## Módulo de data_load

Comportamiento esperado
- El módulo debe de trabajar con DataFrames
- Se debe de guardar la base de datos en el path especificado en el archivo yaml

Resultados obtenidos
- El módulo cumple con el comportamiento esperado


## Módulo de data_split

Comportamiento esperado
- El módulo debe de entregar 3 conjuntos de datos (entrenamiento, validación y prueba) como DataFrames
- Los conjuntos de datos deben de contener a la variable de salida
- Los nombres de las columnas de los conjuntos deben de ser strings
- Se deben de guardar los conjuntos en el path especificado en el archivo yaml

Resultados obtenidos
- El módulo cumple con el comportamiento esperado


## Módulo de features_transform

Comportamiento esperado
- El módulo debe de aplicar correctamente la codificación LabelEncoder() a la variable de salida de todos los conjuntos
- Se debe de aplicar correctamente la codificación OneHotEncoder() y OrdinalEncoder() a las variables de entrada de todos los conjuntos

Resultados obtenidos
- El módulo cumple con el comportamiento esperado


## Módulo de train_model

Comportamiento esperado
- El módulo debe de ajustar correctamente el modelo
- Se debe de guardar el modelo en el path especificado en el archivo yaml

Resultados obtenidos
- El módulo cumple con el comportamiento esperado


## Módulo de evaluate

Comportamiento esperado
- Al aplicarse la validación cruzada, el módulo debe de entregar las puntuaciones correspondientes a cada partición
- EL módulo debe de generar métricas de rendimiento del modelo y una matriz de confusión
- Se deben de guardar las métricas y la imagen de la matriz de confusión en el path especificado en el archivo yaml

Resultados obtenidos
- El módulo cumple con el comportamiento esperado


# Pruebas de integración

Comportamiento esperado
- Se deben de obtener las métricas: accuracy, mse y rmse

Resultados obtenidos
- El módulo cumple con el comportamiento esperado