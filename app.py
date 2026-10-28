from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de aprobación estudiantil",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f6f8fb; border: 1px solid #e5eaf0; padding: 0.9rem; border-radius: 0.8rem;}
    .nota {padding: 0.9rem 1rem; border-left: 4px solid #3973ac; background: #f5f8fc; border-radius: 0.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo_mlp_estudiantes.joblib"


@st.cache_resource(show_spinner="Cargando modelo MLP...")
def cargar_artefacto():
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            "No se encontró modelo_mlp_estudiantes.joblib junto a app.py."
        )
    return joblib.load(MODELO_PATH)


try:
    artefacto = cargar_artefacto()
except Exception as exc:
    st.error(f"No fue posible cargar el modelo: {exc}")
    st.stop()

modelo = artefacto["modelo"]
features = artefacto["features"]
threshold = float(artefacto.get("threshold", 0.5))
etiquetas = artefacto["etiquetas"]
rangos = artefacto["rangos_observados"]
ui = artefacto["configuracion_ui"]

st.title("🎓 Predicción de aprobación estudiantil")
st.caption(
    "Perceptrón Multicapa (MLP) con imputación por mediana y estandarización integradas."
)

st.markdown(
    """
    <div class="nota">
    La aplicación estima la probabilidad de aprobación con seis variables académicas y de hábitos.
    <b>Calificacion_Final no se utiliza</b>, porque es una salida derivada y generaría fuga de información.
    </div>
    """,
    unsafe_allow_html=True,
)

pestana_individual, pestana_lote, pestana_modelo = st.tabs(
    ["Predicción individual", "Predicción por lote", "Rendimiento del modelo"]
)


def predecir(dataframe: pd.DataFrame):
    probabilidades = modelo.predict_proba(dataframe[features])[:, 1]
    clases = (probabilidades >= threshold).astype(int)
    return clases, probabilidades


with pestana_individual:
    st.subheader("Datos del estudiante")
    with st.form("formulario_estudiante"):
        col1, col2 = st.columns(2)

        with col1:
            tareas = st.slider(
                etiquetas["Tareas"],
                min_value=float(ui["Tareas"]["min"]),
                max_value=float(ui["Tareas"]["max"]),
                value=float(round(rangos["Tareas"]["mediana"])),
                step=float(ui["Tareas"]["step"]),
            )
            examenes = st.slider(
                etiquetas["Examenes"],
                min_value=float(ui["Examenes"]["min"]),
                max_value=float(ui["Examenes"]["max"]),
                value=float(round(rangos["Examenes"]["mediana"])),
                step=float(ui["Examenes"]["step"]),
            )
            asistencia = st.slider(
                etiquetas["Asistencia"],
                min_value=float(ui["Asistencia"]["min"]),
                max_value=float(ui["Asistencia"]["max"]),
                value=float(round(rangos["Asistencia"]["mediana"])),
                step=float(ui["Asistencia"]["step"]),
            )

        with col2:
            estudio = st.slider(
                etiquetas["Estudio_Horas"],
                min_value=float(ui["Estudio_Horas"]["min"]),
                max_value=float(ui["Estudio_Horas"]["max"]),
                value=float(round(rangos["Estudio_Horas"]["mediana"] * 2) / 2),
                step=float(ui["Estudio_Horas"]["step"]),
            )
            sueno = st.slider(
                etiquetas["Sueno_Horas"],
                min_value=float(ui["Sueno_Horas"]["min"]),
                max_value=float(ui["Sueno_Horas"]["max"]),
                value=float(round(rangos["Sueno_Horas"]["mediana"] * 2) / 2),
                step=float(ui["Sueno_Horas"]["step"]),
            )
            extracurricular = st.selectbox(
                etiquetas["Extracurriculares"],
                options=[0, 1],
                format_func=lambda x: "Sí" if x == 1 else "No",
            )

        enviar = st.form_submit_button("Generar predicción", type="primary", use_container_width=True)

    if enviar:
        entrada = pd.DataFrame(
            [{
                "Tareas": tareas,
                "Examenes": examenes,
                "Asistencia": asistencia,
                "Estudio_Horas": estudio,
                "Sueno_Horas": sueno,
                "Extracurriculares": extracurricular,
            }]
        )
        clases, probabilidades = predecir(entrada)
        clase = int(clases[0])
        prob = float(probabilidades[0])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Resultado estimado", "Aprobado" if clase == 1 else "Reprobado")
        m2.metric("Probabilidad de aprobar", f"{prob:.1%}")
        m3.metric("Probabilidad de reprobar", f"{1-prob:.1%}")
        st.progress(prob, text="Probabilidad estimada de aprobación")

        if prob >= 0.75:
            st.success("Pronóstico favorable. Mantener los hábitos y el seguimiento académico.")
        elif prob >= threshold:
            st.success("Pronóstico de aprobación, aunque con margen moderado. Conviene reforzar las variables más débiles.")
        elif prob >= 0.35:
            st.warning("Riesgo moderado de reprobación. Se recomienda acompañamiento preventivo y seguimiento cercano.")
        else:
            st.error("Riesgo alto de reprobación. Se recomienda priorizar una intervención académica temprana.")

        st.dataframe(
            entrada.rename(columns=etiquetas).T.rename(columns={0: "Valor suministrado"}),
            use_container_width=True,
        )

with pestana_lote:
    st.subheader("Predicción por archivo CSV")
    st.write("El archivo debe contener estas columnas:", ", ".join(features))

    ejemplo = pd.DataFrame([
        {"Tareas": 78, "Examenes": 72, "Asistencia": 88, "Estudio_Horas": 17, "Sueno_Horas": 7, "Extracurriculares": 1},
        {"Tareas": 45, "Examenes": 38, "Asistencia": 62, "Estudio_Horas": 6, "Sueno_Horas": 6, "Extracurriculares": 0},
    ])
    st.download_button(
        "Descargar plantilla CSV",
        data=ejemplo.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_estudiantes.csv",
        mime="text/csv",
    )

    archivo = st.file_uploader("Cargue el CSV", type=["csv"])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
            faltantes = [c for c in features if c not in lote.columns]
            if faltantes:
                st.error(f"Faltan columnas obligatorias: {faltantes}")
            else:
                datos_modelo = lote[features].copy()
                for columna in features:
                    datos_modelo[columna] = pd.to_numeric(datos_modelo[columna], errors="coerce")
                clases, probabilidades = predecir(datos_modelo)
                salida = lote.copy()
                salida["Prediccion"] = np.where(clases == 1, "Aprobado", "Reprobado")
                salida["Probabilidad_Aprobado"] = probabilidades
                salida["Probabilidad_Reprobado"] = 1 - probabilidades

                st.success(f"Se procesaron {len(salida)} registros.")
                st.dataframe(salida, use_container_width=True, hide_index=True)
                st.download_button(
                    "Descargar resultados",
                    data=salida.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_estudiantes.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"No fue posible procesar el archivo: {exc}")

with pestana_modelo:
    st.subheader("Evaluación sobre el conjunto de prueba independiente")
    metricas = artefacto["metricas_prueba_independiente"]
    cols = st.columns(5)
    for col, nombre in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
        col.metric(nombre, f"{metricas[nombre]:.3f}")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("#### Matriz de confusión")
        cm = np.asarray(artefacto["matriz_confusion"])
        cm_df = pd.DataFrame(
            cm,
            index=["Real: Reprobado", "Real: Aprobado"],
            columns=["Predicho: Reprobado", "Predicho: Aprobado"],
        )
        st.dataframe(cm_df, use_container_width=True)

        balance = pd.Series(artefacto["balance_clases"], name="Registros")
        st.markdown("#### Balance de clases")
        st.bar_chart(balance)

    with derecha:
        st.markdown("#### Importancia por permutación")
        importancia = (
            pd.Series(artefacto.get("importancia_permutacion", {}), name="Importancia")
            .sort_values(ascending=True)
        )
        if importancia.empty:
            st.caption("La importancia por permutación no está disponible para este artefacto.")
        else:
            st.bar_chart(importancia)

        st.markdown("#### Arquitectura seleccionada")
        st.json(artefacto["mejores_parametros"])

    st.info(
        "Estos resultados describen un dataset sintético. Antes de tomar decisiones reales se requiere "
        "validación externa, monitoreo de sesgos y revisión humana."
    )

st.divider()
st.caption(
    f"Modelo generado el {artefacto.get('fecha_generacion', 'N/D')} · "
    f"Umbral de clasificación: {threshold:.2f} · Uso académico/demostrativo"
)
