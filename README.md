# Predicción de aprobación estudiantil con MLP

Aplicación Streamlit generada desde el notebook del taller. El modelo recibe seis variables, aplica imputación por mediana, estandarización y una red neuronal MLP.

## Contenido

- `app.py`: interfaz Streamlit para predicción individual, por lotes y consulta de métricas.
- `modelo_mlp_estudiantes.joblib`: Pipeline final listo para inferencia.
- `train_model.py`: script para reentrenar el modelo.
- `dataset_estudiantes_balanceado.csv`: datos del taller.
- `ejemplo_entrada.csv`: plantilla de carga por lotes.
- `requirements.txt`: dependencias de Python.
- `.streamlit/config.toml`: tema visual.
- `PYTHON_VERSION.txt`: versión de Python usada para generar el artefacto.
- `LINK_ENDPOINT.txt`: archivo para registrar la URL publicada.

## Ejecutar localmente

```bash
python -m venv .venv
```

Active el entorno virtual y ejecute:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Descomprima el ZIP y suba **el contenido de la carpeta** a un repositorio de GitHub.
2. En Streamlit Community Cloud, cree una aplicación nueva y seleccione el repositorio, la rama y `app.py` como archivo principal.
3. En la configuración avanzada, seleccione la misma versión mayor y menor de Python indicada en `PYTHON_VERSION.txt`.
4. Despliegue la aplicación.
5. Copie la URL `https://...streamlit.app` en `LINK_ENDPOINT.txt` y en el informe de entrega.

## Consideraciones metodológicas

- `Calificacion_Final` no se utiliza como entrada porque es una salida derivada y causaría fuga de información.
- Las métricas guardadas corresponden a un conjunto de prueba independiente antes de reentrenar el modelo final con todos los registros.
- Los datos son sintéticos. El resultado debe entenderse como demostración académica y no como decisión definitiva sobre una persona.
