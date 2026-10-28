from pathlib import Path

import joblib
import numpy as np
import altair as alt
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

        enviar = st.form_submit_button("Generar predicción", type="primary", width="stretch")

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
            width="stretch",
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
                st.dataframe(salida, width="stretch", hide_index=True)
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
    st.caption(
        f"{artefacto.get('tamano_train', 'N/D')} registros de entrenamiento y "
        f"{artefacto.get('tamano_test', 'N/D')} de prueba (división 80/20 estratificada). "
        "El set de prueba no se usó para elegir la arquitectura."
    )
    metricas = artefacto["metricas_prueba_independiente"]
    cols = st.columns(5)
    for col, nombre in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
        col.metric(nombre, f"{metricas[nombre]:.3f}")

    izquierda, derecha = st.columns(2)
    with izquierda:
        st.markdown("#### Matriz de confusión")
        cm = np.asarray(artefacto["matriz_confusion"])
        clases_txt = ["Reprobado", "Aprobado"]
        cm_largo = pd.DataFrame(
            [
                {"Real": clases_txt[i], "Predicho": clases_txt[j], "Cantidad": int(cm[i, j])}
                for i in range(2)
                for j in range(2)
            ]
        )
        base_cm = alt.Chart(cm_largo).encode(
            x=alt.X("Predicho:N", sort=clases_txt, title="Predicción", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Real:N", sort=clases_txt, title="Valor real"),
        )
        heatmap = base_cm.mark_rect().encode(
            color=alt.Color("Cantidad:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Real", "Predicho", "Cantidad"],
        )
        textos = base_cm.mark_text(fontSize=22, fontWeight="bold").encode(
            text="Cantidad:Q",
            color=alt.condition(
                alt.datum.Cantidad > cm.max() / 2, alt.value("white"), alt.value("#17202A")
            ),
        )
        st.altair_chart((heatmap + textos).properties(height=280), width="stretch")
        vn, fp, fn, vp = cm.ravel()
        st.caption(
            f"Aciertos: {vn + vp} de {cm.sum()} · Falsos positivos (reprobado predicho como aprobado): {fp} · "
            f"Falsos negativos (aprobado predicho como reprobado): {fn}"
        )

        st.markdown("#### Entrenamiento vs. prueba")
        entrenamiento = artefacto.get("metricas_entrenamiento")
        if entrenamiento:
            comparacion = pd.DataFrame(
                {"Entrenamiento": entrenamiento, "Prueba": metricas}
            ).map(lambda v: f"{v:.3f}")
            st.dataframe(comparacion, width="stretch")
            st.caption("Una diferencia pequeña entre ambas columnas indica que el modelo no está sobreajustado.")

        balance = pd.Series(artefacto["balance_clases"], name="Registros")
        st.markdown("#### Balance de clases")
        st.bar_chart(balance, height=220)

    with derecha:
        st.markdown("#### Importancia por permutación")
        importancia = (
            pd.Series(artefacto.get("importancia_permutacion", {}), name="Importancia")
            .sort_values(ascending=True)
        )
        if importancia.empty:
            st.caption("La importancia por permutación no está disponible para este artefacto.")
        else:
            imp_df = importancia.rename_axis("Variable").reset_index()
            grafico_imp = alt.Chart(imp_df).mark_bar(color="#3973AC").encode(
                x=alt.X("Importancia:Q", title="Caída del F1"),
                y=alt.Y("Variable:N", sort="-x", title=None),
                tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".3f")],
            ).properties(height=230)
            st.altair_chart(grafico_imp, width="stretch")
            st.caption("Cuánto cae el F1 en prueba al desordenar cada variable.")

        st.markdown("#### Arquitectura seleccionada")
        mejores = artefacto["mejores_parametros"]
        f1_cv = artefacto.get("f1_cv")
        st.write(
            f"Capas ocultas: **{tuple(mejores['hidden_layer_sizes'])}** · "
            f"Activación: **{mejores['activation']}** · alpha: **{mejores['alpha']}**"
        )
        if f1_cv is not None:
            st.write(
                f"F1 promedio en validación cruzada (5 folds): **{f1_cv:.3f} ± {artefacto.get('f1_cv_std', 0):.3f}** "
                f"({artefacto.get('combinaciones_evaluadas', 'N/D')} combinaciones evaluadas)"
            )
        top = artefacto.get("top_busqueda")
        if top:
            with st.expander("Mejores 10 combinaciones de la búsqueda"):
                st.dataframe(pd.DataFrame(top), width="stretch", hide_index=True)

        estabilidad = artefacto.get("estabilidad_semillas")
        if estabilidad:
            st.markdown("#### Estabilidad")
            st.write(
                f"Accuracy en prueba con 10 semillas distintas: **{estabilidad['media']:.3f} ± "
                f"{estabilidad['desviacion']:.3f}**"
            )

    modelo_base = artefacto.get("modelo_base")
    if modelo_base:
        st.markdown("#### Comparación con un modelo base")
        tabla_base = pd.DataFrame(
            {"MLP": metricas, modelo_base["nombre"]: modelo_base["metricas"]}
        ).map(lambda v: f"{v:.3f}")
        st.dataframe(tabla_base, width="stretch")

    analisis = artefacto.get("analisis_calificacion_final")
    if analisis:
        st.markdown("#### Hallazgo sobre los datos")
        coef = analisis["coeficientes"]
        formula = " + ".join(f"{v:g}·{k}" for k, v in coef.items())
        st.markdown(
            f"`Calificacion_Final` es una combinación lineal exacta de las seis variables "
            f"(R² = {analisis['r2_combinacion_lineal']:.3f}):\n\n"
            f"`Calificacion_Final = {formula}`\n\n"
            f"y `Aprobado` vale 1 cuando esa nota supera la mediana "
            f"(≈ {analisis['mediana_calificacion_final']:.2f}). Por eso las clases se separan con una frontera "
            "lineal y la regresión logística alcanza el mismo desempeño que el MLP."
        )

    st.info(
        "Estos resultados describen un dataset sintético. Antes de tomar decisiones reales se requiere "
        "validación externa, monitoreo de sesgos y revisión humana."
    )

st.divider()
st.caption(
    f"Modelo generado el {artefacto.get('fecha_generacion', 'N/D')} · "
    f"Umbral de clasificación: {threshold:.2f} · Uso académico/demostrativo"
)
