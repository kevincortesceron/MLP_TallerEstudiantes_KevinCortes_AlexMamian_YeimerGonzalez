"""Entrena el MLP del taller y guarda el artefacto que usa la app Streamlit.

Pasos:
1. Carga del dataset y verificación de nulos, duplicados y balance de clases.
2. División 80/20 estratificada (el 20 % de prueba no se toca hasta el final).
3. Búsqueda de la arquitectura con GridSearchCV (validación cruzada de 5 folds)
   sobre un Pipeline: imputación por mediana -> estandarización -> MLP.
4. Evaluación en el set de prueba independiente: matriz de confusión,
   Precision, Recall, F1-Score, Accuracy y ROC-AUC.
5. Comparación con una regresión logística (modelo base) y prueba de estabilidad
   del MLP con distintas semillas.
6. Reentrenamiento final con todos los datos y guardado del artefacto.
"""
from datetime import date
from pathlib import Path
import platform

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset_estudiantes_balanceado.csv"
OUTPUT = BASE_DIR / "modelo_mlp_estudiantes.joblib"
FIGURA_CM = BASE_DIR / "matriz_confusion.png"
FEATURES = ["Tareas", "Examenes", "Asistencia", "Estudio_Horas", "Sueno_Horas", "Extracurriculares"]
TARGET = "Aprobado"
DERIVADA = "Calificacion_Final"

# Espacio de búsqueda de la arquitectura del MLP.
PARAM_GRID = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],
}


def crear_pipeline(clasificador):
    """Imputación y escalado dentro del Pipeline: solo aprenden del set de entrenamiento."""
    return Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
        ("mlp", clasificador),
    ])


def crear_mlp(random_state=SEED):
    return MLPClassifier(
        learning_rate_init=0.005,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.20,
        n_iter_no_change=40,
        random_state=random_state,
    )


def metricas(modelo, X, y):
    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1-Score": float(f1_score(y, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y, prob)),
    }, confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist()


def guardar_matriz_confusion(cm, ruta):
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=["Reprobado", "Aprobado"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión (set de prueba)")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def analizar_variable_derivada(df):
    """Verifica cómo se construyó Calificacion_Final y la clase Aprobado."""
    datos = df.dropna(subset=FEATURES + [DERIVADA])
    reg = LinearRegression().fit(datos[FEATURES], datos[DERIVADA])
    umbral = float(datos.loc[datos[TARGET] == 1, DERIVADA].min())
    return {
        "r2_combinacion_lineal": float(reg.score(datos[FEATURES], datos[DERIVADA])),
        "coeficientes": {f: round(float(c), 4) for f, c in zip(FEATURES, reg.coef_)},
        "intercepto": round(float(reg.intercept_), 4),
        "mediana_calificacion_final": float(datos[DERIVADA].median()),
        "umbral_aprobado_observado": umbral,
        "max_calificacion_final": float(datos[DERIVADA].max()),
        "min_calificacion_final": float(datos[DERIVADA].min()),
    }


def main():
    df_crudo = pd.read_csv(DATASET)
    duplicados = int(df_crudo.duplicated().sum())
    df = df_crudo.drop_duplicates().dropna(subset=[TARGET])
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    nulos = {c: int(v) for c, v in X.isna().sum().items()}

    print(f"Registros: {len(df)} | duplicados eliminados: {duplicados}")
    print("Nulos por columna:", nulos)
    print("Balance de clases:", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # 1) Búsqueda de la arquitectura con validación cruzada (solo set de entrenamiento).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    busqueda = GridSearchCV(
        crear_pipeline(crear_mlp()), PARAM_GRID, scoring="f1", cv=cv, n_jobs=-1, refit=True
    )
    busqueda.fit(X_train, y_train)
    modelo_eval = busqueda.best_estimator_
    mejores = {k.replace("mlp__", ""): v for k, v in busqueda.best_params_.items()}
    i_mejor = busqueda.best_index_
    print("Mejores parámetros:", mejores)
    print(f"F1 en CV: {busqueda.best_score_:.4f} ± {busqueda.cv_results_['std_test_score'][i_mejor]:.4f}")

    resultados = pd.DataFrame(busqueda.cv_results_)
    top = (
        resultados.sort_values("rank_test_score")
        .head(10)
        .assign(
            Capas=lambda d: d["param_mlp__hidden_layer_sizes"].astype(str),
            Activacion=lambda d: d["param_mlp__activation"],
            Alpha=lambda d: d["param_mlp__alpha"].astype(float),
            F1_CV=lambda d: d["mean_test_score"].round(4),
            Desv=lambda d: d["std_test_score"].round(4),
        )[["Capas", "Activacion", "Alpha", "F1_CV", "Desv"]]
    )

    # 2) Evaluación en el set de prueba independiente.
    metricas_train, _ = metricas(modelo_eval, X_train, y_train)
    metricas_test, cm = metricas(modelo_eval, X_test, y_test)
    print("Métricas de entrenamiento:", metricas_train)
    print("Métricas de prueba:", metricas_test)
    print("Matriz de confusión:", cm)
    print(classification_report(y_test, modelo_eval.predict(X_test),
                                target_names=["Reprobado", "Aprobado"], digits=3))
    guardar_matriz_confusion(cm, FIGURA_CM)

    # 3) Modelo base: regresión logística con el mismo preprocesamiento.
    base = crear_pipeline(LogisticRegression(max_iter=5000, random_state=SEED))
    base.fit(X_train, y_train)
    metricas_base, cm_base = metricas(base, X_test, y_test)
    print("Regresión logística (base):", metricas_base)

    # 4) Estabilidad: la mejor arquitectura con 10 semillas distintas.
    acc_semillas = []
    for semilla in range(10):
        m = clone(modelo_eval).set_params(mlp__random_state=semilla).fit(X_train, y_train)
        acc_semillas.append(float(accuracy_score(y_test, m.predict(X_test))))
    print(f"Accuracy con 10 semillas: {np.mean(acc_semillas):.4f} ± {np.std(acc_semillas):.4f}")

    # 5) Importancia por permutación sobre el set de prueba.
    perm = permutation_importance(
        modelo_eval, X_test, y_test, scoring="f1", n_repeats=30, random_state=SEED, n_jobs=-1
    )
    importancias = {f: float(v) for f, v in zip(FEATURES, perm.importances_mean)}

    analisis = analizar_variable_derivada(df)
    print("Análisis de Calificacion_Final:", analisis)

    # 6) Reentrenamiento final con todos los datos (copia nueva, sin alterar modelo_eval).
    modelo_final = clone(modelo_eval).fit(X, y)

    artefacto = {
        "modelo": modelo_final,
        "features": FEATURES,
        "target": TARGET,
        "threshold": 0.50,
        "etiquetas": {
            "Tareas": "Calificación de tareas",
            "Examenes": "Calificación de exámenes",
            "Asistencia": "Asistencia (%)",
            "Estudio_Horas": "Horas de estudio por semana",
            "Sueno_Horas": "Horas de sueño por día",
            "Extracurriculares": "Participa en actividades extracurriculares",
        },
        "configuracion_ui": {
            "Tareas": {"min": 0.0, "max": 100.0, "step": 1.0},
            "Examenes": {"min": 0.0, "max": 100.0, "step": 1.0},
            "Asistencia": {"min": 0.0, "max": 100.0, "step": 1.0},
            "Estudio_Horas": {"min": 0.0, "max": 40.0, "step": 0.5},
            "Sueno_Horas": {"min": 0.0, "max": 12.0, "step": 0.5},
            "Extracurriculares": {"min": 0, "max": 1, "step": 1},
        },
        "rangos_observados": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()), "mediana": float(df[c].median())}
            for c in FEATURES
        },
        "metricas_prueba_independiente": metricas_test,
        "metricas_entrenamiento": metricas_train,
        "matriz_confusion": cm,
        "mejores_parametros": {**mejores, "hidden_layer_sizes": list(mejores["hidden_layer_sizes"])},
        "f1_cv": float(busqueda.best_score_),
        "f1_cv_std": float(busqueda.cv_results_["std_test_score"][i_mejor]),
        "top_busqueda": top.to_dict(orient="records"),
        "combinaciones_evaluadas": int(len(resultados)),
        "modelo_base": {
            "nombre": "Regresión logística",
            "metricas": metricas_base,
            "matriz_confusion": cm_base,
        },
        "estabilidad_semillas": {
            "accuracies": acc_semillas,
            "media": float(np.mean(acc_semillas)),
            "desviacion": float(np.std(acc_semillas)),
        },
        "balance_clases": {"Reprobado": int((y == 0).sum()), "Aprobado": int((y == 1).sum())},
        "nulos_por_columna": nulos,
        "duplicados_eliminados": duplicados,
        "importancia_permutacion": importancias,
        "analisis_calificacion_final": analisis,
        "filas_dataset": int(len(df)),
        "tamano_train": int(len(X_train)),
        "tamano_test": int(len(X_test)),
        "fecha_generacion": date.today().isoformat(),
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "nota_metodologica": "Calificacion_Final se excluye porque causaría fuga de información.",
    }
    joblib.dump(artefacto, OUTPUT, compress=3)
    print(f"Modelo guardado en {OUTPUT}")
    print(f"Matriz de confusión guardada en {FIGURA_CM}")


if __name__ == "__main__":
    main()
