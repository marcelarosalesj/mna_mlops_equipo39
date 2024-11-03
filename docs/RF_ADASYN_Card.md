# 1. Propósito del Modelo
## 1.1 Descripción
El modelo de predicción de calificaciones busca identificar estudiantes con alto potencial académico y estudiantes en riesgo de reprobar. Esto permitirá a la institución educativa implementar programas de apoyo específicos, como becas para alumnos destacados y asesorías o intervenciones adicionales para estudiantes en riesgo académico.

## 1.2 Datos de Entrada
El modelo utiliza datos provenientes de una encuesta que abarca tres áreas principales:

- Características personales: género, tipo de bachillerato, tipo de beca, actividades extracurriculares, etc.
- Contexto familiar: nivel educativo de los padres, ocupación, estado civil de los padres, número de hermanos y salario familiar total.
- Hábitos de estudio: asistencia a clases, horas de estudio semanal, GPA acumulado, hábitos de lectura, entre otros.
# 2. Métricas de Rendimiento
## 2.1 Principales Métricas de Evaluación
- Exactitud (accuracy)
- F1 Score
- Precisión (precision)
- Recall
- Matriz de Confusión
- Reporte de Clasificación

## 2.2 Resultados de las Métricas
- Accuracy: 0.75
- Recall: 0.75
- Train Accuracy: 1.0 (indicador de posible sobreajuste en el modelo)
- Precisión: 0.76
- F1 Score: 0.748

# 3. Conjunto de Datos
## 3.1 Descripción del Dataset
El conjunto de datos fue recolectado en 2019, incluyendo estudiantes de la Facultad de Ingeniería y de la Facultad de Ciencias de la Educación. Los datos están disponibles en Kaggle con fines educativos.

## 3.2 Estadísticas del Dataset
- Número de Registros: 145
- Número de Características: 31
- Preprocesamiento: Todas las variables están codificadas numéricamente y no contienen valores faltantes.
## 3.3 Transformaciones de Datos
Aumento de Datos: Dado el bajo número de registros, se aplicó la técnica ADASYN para balancear las clases y aumentar el volumen de datos de entrenamiento.
Transformación de Variables Categóricas:
- One-Hot Encoding para variables categóricas nominales.
- Ordinal Encoding para variables categóricas ordinales.
# 4. Consideraciones Éticas y de Sesgo
## 4.1 Evaluación de Sesgo
El modelo incluye características demográficas (género, edad y salario), que pueden influir en los resultados y presentar sesgos. Actualmente, no se ha realizado una evaluación de sesgo.

## 4.2 Mitigación del Sesgo
Por el momento, no se han implementado medidas de mitigación de sesgo. Se recomienda incluir una evaluación y técnicas de mitigación en futuras versiones del modelo.

# 5. Limitaciones del Modelo
## 5.1 Limitaciones Principales
Volumen de Datos: Con un total de solo 145 registros, la representatividad de los datos es limitada, lo que afecta la precisión del modelo en ciertas clases.
Desbalanceo de Clases: Las clases 1 y 2 (que representan estudiantes con calificaciones bajas) son difíciles de predecir debido a su menor representación en el dataset.
## 5.2 Uso Recomendado y No Recomendado
Uso Recomendado: Este modelo es adecuado para identificar a estudiantes en riesgo de reprobar (clase 0).
Uso No Recomendado: En caso de implementar acciones para estudiantes en las clases "DD" o "DC" (clases 1 y 2), se recomienda mejorar la capacidad del modelo en estas categorías antes de su despliegue.
# 6. Historial de Versiones y Cambios
- Versión Actual: Incluye la técnica de resampling ADASYN para balancear las clases y mejorar el rendimiento del modelo en categorías minoritarias.
- Versiones Anteriores: Las versiones previas no incluían ADASYN, lo que resultaba en un rendimiento deficiente al tratar de predecir clases desbalanceadas, especialmente en las clases 1 y 2.