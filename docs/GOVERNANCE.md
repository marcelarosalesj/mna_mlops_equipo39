# Gobernanza

## 1. Objetivo de Gobernanza

Este documento define las políticas y estándares de gobernanza para asegurar que el código del proyecto de predicción de calificaciones mantenga altos niveles de calidad, seguridad y cumplimiento de buenas prácticas.

## 2. Estándares de Código

Para asegurar la consistencia y calidad del código, se implementan las siguientes herramientas de linting y formato:

- *ruff*: Usado para verificar errores de sintaxis, linting del código y formato de python y notebooks. [ruff repositorio](https://github.com/astral-sh/ruff)

### 2.1 Configuración de ruff

- Configuramos ruff con el target `py39`.

### 2.2 Pre-commit Hooks

Para automatizar el proceso de linting y formato, se configura un github actions que ejecuta ruff, pruebas unitarias y pruebas de integración de forma automatizada para todos los pull requests. Esto evita que el código que no cumple con los estándares sea agregado al repositorio.


## 3. Proceso de Revisión de Código (Code Review)

Todo cambio en el código debe pasar por una revisión antes de fusionarse en la rama principal. Los pasos son los siguientes:

1. *Creación de Pull Request*: Cualquier cambio debe ser subido a GitHub mediante un Pull Request, hacer push al branch `main` está bloqueado.
2. *Asignación de Revisores*: Se asignan al menos uno o dos miembros del equipo para revisar el Pull Request.
3. *Revisión de Estándares*: Los revisores verifican que el código cumpla con las reglas de linting y formato establecidas, y que se sigan las buenas prácticas de Machine Learning.
4. *Aprobación*: Una vez que el revisor apruebe el Pull Request, se puede fusionar con la rama principal `main`.

## 4. Beneficios de los Estándares de Código y Revisión

Implementar estos estándares de código y revisiones asegura:

- *Consistencia*: El código mantiene un formato uniforme, lo que facilita su comprensión y mantenimiento.
- *Calidad*: La revisión de código ayuda a detectar errores y mejorar la calidad general del proyecto.
- *Colaboración*: Las revisiones promueven la colaboración y el aprendizaje en el equipo, asegurando que todos los cambios sean revisados y aprobados.
- *Cumplimiento*: Asegura que el proyecto siga buenas prácticas y estándares de la industria.
