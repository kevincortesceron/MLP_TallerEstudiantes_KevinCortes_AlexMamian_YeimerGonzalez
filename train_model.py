from pathlib import Path
from datetime import date
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset_estudiantes_balanceado.csv"
OUTPUT = BASE_DIR / "modelo_mlp_estudiantes.joblib"
FEATURES = ["Tareas", "Examenes", "Asistencia", "Estudio_Horas", "Sueno_Horas", "Extracurriculares"]
TARGET = "Aprobado"


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


def main():
    df = pd.read_csv(DATASET).drop_duplicates().dropna(subset=[TARGET])
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    modelo_eval = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='tanh',
            alpha=0.0001,
            learning_rate_init=0.005,
            random_state=SEED,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=40,
        )),
    ])
    modelo_eval.fit(X_train, y_train)
    metricas_train, _ = metricas(modelo_eval, X_train, y_train)
    metricas_test, cm = metricas(modelo_eval, X_test, y_test)

    perm = permutation_importance(
        modelo_eval, X_test, y_test, scoring="f1", n_repeats=30, random_state=SEED, n_jobs=-1
    )
    importancias = {
        feature: float(value) for feature, value in zip(FEATURES, perm.importances_mean)
    }

    # Reentrenamiento final con todos los datos.
    modelo_final = modelo_eval
    modelo_final.fit(X, y)

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
        "mejores_parametros": {
            "hidden_layer_sizes": (32, 16),
            "activation": 'tanh',
            "alpha": 0.0001,
            "learning_rate_init": 0.005,
        },
        "f1_cv": None,
        "balance_clases": {
            "Reprobado": int((y == 0).sum()),
            "Aprobado": int((y == 1).sum()),
        },
        "importancia_permutacion": importancias,
        "filas_dataset": int(len(df)),
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
    print("Métricas de prueba:", metricas_test)


if __name__ == "__main__":
    main()
