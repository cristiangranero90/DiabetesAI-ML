# DiabetesAI-ML

Descripción

DiabetesAI-ML es un proyecto de aprendizaje automático orientado a analizar y predecir el riesgo de diabetes usando técnicas de clasificación supervisada. Incluye preprocesamiento de datos, entrenamiento de modelos, evaluación y generación de métricas para comparar rendimiento.

Objetivo

Construir modelos reproducibles que permitan predecir la probabilidad de que un paciente presente diabetes, facilitando análisis y experimentación con distintos algoritmos y características.

Datos

Los datos provienen de conjuntos públicos (CSV) con variables clínicas y demográficas. El repositorio contiene scripts para carga, limpieza y transformación de las tablas antes del entrenamiento.

Estructura del repositorio

- data/: archivos de datos y scripts de descarga
- notebooks/: notebooks exploratorios y de experimentación
- src/: código fuente para preprocesamiento, modelos y utilidades
- models/: artefactos de modelos entrenados y checkpoints
- results/: métricas, gráficos y reportes de evaluación

Requisitos

- Python 3.8+ (preferible 3.10)
- dependencias en requirements.txt (pandas, scikit-learn, joblib, matplotlib, seaborn, etc.)

Uso rápido

1. Crear entorno virtual: `python -m venv .venv` y activarlo.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Ejecutar preprocessing: `python src/preprocess.py --input data/raw.csv --output data/clean.csv`.
4. Entrenar modelo: `python src/train.py --data data/clean.csv --output models/`.
5. Evaluar: `python src/evaluate.py --model models/model.joblib --data data/test.csv`.

Resultados y evaluación

Se generan métricas estándar (accuracy, precision, recall, F1, AUC) y gráficos de validación cruzada. Los resultados se guardan en `results/`.

Contribuir

Abrir issues o pull requests con mejoras en datos, modelos o documentación. Seguir buenas prácticas: tests, reproducibilidad y comentarios claros.

Contacto

Cristian Granero (autor del repositorio).