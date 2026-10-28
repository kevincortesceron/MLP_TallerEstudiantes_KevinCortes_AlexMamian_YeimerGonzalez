# Análisis y conclusiones — MLP para predecir la aprobación estudiantil

Todos los números de este documento salen de `train_model.py` (semilla 42). Se reproducen al ejecutar `python train_model.py`.

## 1. Análisis de los datos

| Aspecto | Resultado |
|---|---|
| Registros | 600 (6 variables de entrada, `Calificacion_Final` y `Aprobado`) |
| Duplicados | 0 |
| Valores nulos | 0 en todas las columnas |
| Balance de clases | 300 aprobados y 300 reprobados (50 % / 50 %) |

- **Balance de clases (objetivo 1):** el dataset ya cumple la proporción 50/50, así que no hizo falta sobremuestrear ni submuestrear. Para conservar esa proporción, la división entrenamiento/prueba es estratificada.
- **Imputación por mediana:** está implementada dentro del `Pipeline` como pide el taller, pero en este dataset no modifica ningún valor porque no hay nulos. Igual queda lista para cuando se carguen archivos por lote con datos faltantes, y como está dentro del `Pipeline`, la mediana se calcula solo con el set de entrenamiento.
- **Estandarización:** `StandardScaler` deja cada variable con media 0 y varianza 1. Es necesaria porque las escalas son muy distintas (por ejemplo, 0–100 en las notas y 4–9 en las horas de sueño), y sin escalar el descenso del gradiente del MLP converge mal.

### Cómo se construyó la variable objetivo

Una regresión lineal de `Calificacion_Final` sobre las seis variables da R² = 1.000. Es decir, la nota final es una combinación lineal exacta:

```
Calificacion_Final = 0.3·Tareas + 0.4·Examenes + 0.1·Asistencia
                   + 1.5·Estudio_Horas + 0.5·Sueno_Horas + 2·Extracurriculares
```

y `Aprobado = 1` cuando esa nota supera la mediana (≈ 74.67). Esto tiene tres consecuencias:

1. **`Calificacion_Final` no puede ser una entrada.** Con ella el modelo solo tendría que comparar un número contra un umbral, y obtendría 100 % sin aprender nada. Es fuga de información, por eso se excluye.
2. **Las clases se separan con una frontera lineal (un hiperplano).** Eso explica por qué todos los modelos alcanzan una exactitud cercana al 99 %.
3. **La escala de la nota no es coherente.** `Calificacion_Final` va de 37.5 a 110.8, así que supera el máximo de 100 que se esperaría de una calificación. Es un detalle del generador sintético que vale la pena señalar.

## 2. Diseño del MLP (objetivo 2)

La arquitectura se eligió con `GridSearchCV` y validación cruzada estratificada de 5 folds, usando **solo el set de entrenamiento** (480 registros) y F1 como métrica. Se evaluaron 30 combinaciones:

- Capas ocultas: (16), (32), (32, 16), (64, 32), (64, 32, 16)
- Activación: ReLU, tanh
- Regularización L2 (`alpha`): 0.0001, 0.001, 0.01

Parámetros fijos: `learning_rate_init = 0.005` y `early_stopping` con 20 % de validación interna y paciencia de 40 épocas, para cortar el entrenamiento antes de sobreajustar.

**Arquitectura seleccionada:** 3 capas ocultas (64, 32, 16), activación **tanh** y alpha = 0.0001. Su F1 en validación cruzada fue **0.983 ± 0.011**.

Observaciones de la búsqueda:
- **tanh superó a ReLU** en las mejores configuraciones. Con pocos datos y una frontera suave, la salida acotada y centrada en cero de tanh funciona mejor.
- **`alpha` casi no influyó:** las tres opciones empataron en la mejor arquitectura. La regularización importa poco porque el problema es sencillo y el early stopping ya controla el sobreajuste.
- **Las diferencias entre arquitecturas son pequeñas** (alrededor de 1 punto de F1). Una red más grande no aporta mucho cuando la frontera es lineal.

## 3. Evaluación en el set de prueba independiente (objetivo 3)

Se usaron 120 registros (20 %) que no intervinieron ni en el entrenamiento ni en la búsqueda.

**Matriz de confusión** (también en `matriz_confusion.png` y en la app):

|  | Predicho: Reprobado | Predicho: Aprobado |
|---|---|---|
| **Real: Reprobado** | 59 (VN) | 1 (FP) |
| **Real: Aprobado** | 0 (FN) | 60 (VP) |

| Métrica | Valor | Interpretación |
|---|---|---|
| Accuracy | 0.992 | 119 de 120 aciertos |
| Precision | 0.984 | De los 61 que el modelo marcó como aprobados, 60 lo eran |
| Recall | 1.000 | Detectó a los 60 aprobados reales |
| F1-Score | 0.992 | Equilibrio entre precisión y recall |
| ROC-AUC | 1.000 | Ordena perfectamente por probabilidad a aprobados y reprobados |

**Sobreajuste:** el modelo obtiene 1.000 en entrenamiento y 0.992 en prueba. La diferencia es de menos de 1 punto, así que generaliza bien.

**Estabilidad:** con la misma arquitectura y 10 semillas distintas, la exactitud en prueba fue de 0.992 ± 0.004. El resultado no depende de una inicialización afortunada.

### Qué métrica importa en este problema

Con clases balanceadas, el accuracy es una métrica confiable. Sin embargo, el caso de uso es **detectar a tiempo a quien va a reprobar** para intervenir, así que el error más costoso es predecir "aprobado" para alguien que en realidad reprueba: el **falso positivo**, si "Aprobado" es la clase positiva. Ese estudiante se queda sin el apoyo que necesitaba. Por eso conviene mirar:

- **El recall de la clase Reprobado: 59/60 = 0.983.** Hubo un solo estudiante en riesgo que el modelo no detectó.
- **La precision de la clase Aprobado: 0.984**, que mide lo mismo desde la otra clase.

Si en la práctica se quisiera minimizar ese error, se podría subir el umbral de decisión por encima de 0.50. Así se marcaría a más estudiantes como "en riesgo", a cambio de algunas alertas innecesarias.

## 4. Comparación con un modelo base

| Métrica | MLP (64, 32, 16) | Regresión logística |
|---|---|---|
| Accuracy | 0.992 | 0.992 |
| Precision | 0.984 | 0.984 |
| Recall | 1.000 | 1.000 |
| F1-Score | 0.992 | 0.992 |
| ROC-AUC | 1.000 | 0.999 |

Las dos fallan exactamente en el mismo registro. Era de esperarse: como la clase depende de una combinación lineal de las entradas, una regresión logística (equivalente a una sola neurona con activación sigmoide) ya tiene la capacidad necesaria. **El MLP cumple el objetivo del taller, pero en este dataset no aporta ventaja sobre un modelo lineal.** Su capacidad para aprender relaciones no lineales sería útil con datos reales, donde la relación entre hábitos y aprobación difícilmente es una suma ponderada.

## 5. Importancia de las variables

Importancia por permutación en el set de prueba, medida como la caída del F1 al desordenar cada variable:

| Variable | Importancia | Coeficiente × desviación estándar en la fórmula |
|---|---|---|
| Estudio_Horas | 0.282 | 1.5 × 6.50 = 9.75 |
| Examenes | 0.251 | 0.4 × 23.28 = 9.31 |
| Tareas | 0.224 | 0.3 × 20.44 = 6.13 |
| Asistencia | 0.078 | 0.1 × 17.14 = 1.71 |
| Extracurriculares | 0.024 | 2.0 × 0.49 = 0.98 |
| Sueno_Horas | 0.018 | 0.5 × 1.47 = 0.74 |

El orden coincide exactamente con el peso real de cada variable en la fórmula. Es decir, **el MLP aprendió la regla que generó los datos**, no un patrón espurio. Las horas de estudio, los exámenes y las tareas explican casi toda la decisión; el sueño y las actividades extracurriculares casi no influyen.

## 6. Conclusiones

1. El MLP alcanza 99.2 % de exactitud y F1 de 0.992 en datos que no vio, sin sobreajuste y con resultados estables entre semillas.
2. El preprocesamiento va dentro de un `Pipeline` (imputación por mediana y estandarización), así que no hay fuga del set de prueba, y la app aplica exactamente las mismas transformaciones al predecir.
3. Excluir `Calificacion_Final` es obligatorio: es una función exacta de las entradas y de ella sale directamente la etiqueta.
4. Las métricas tan altas se explican por cómo se generaron los datos (frontera lineal), no por una superioridad del MLP. Lo demuestra la regresión logística, que obtiene el mismo resultado.
5. Para el uso real (alertas tempranas de reprobación), la métrica clave es el recall de la clase Reprobado, que puede subirse ajustando el umbral de decisión.
6. Al ser datos sintéticos, el modelo sirve como demostración académica. Para usarlo con estudiantes reales haría falta validarlo con datos reales, revisar sesgos y mantener revisión humana en las decisiones.
