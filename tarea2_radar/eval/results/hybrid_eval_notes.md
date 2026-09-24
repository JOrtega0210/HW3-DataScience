# Notas de evaluación — RAG híbrido (Tarea 2, Fase 3)

Generado con `python eval/run_hybrid_eval.py` sobre `hybrid_eval_set.json` (12
preguntas con proceso relevante conocido, sobre el índice real de 20,866
procesos). Ninguna métrica de esta fase llama al LLM (solo retrieval).

## Recall@k

| Métrica | Valor |
|---|---|
| Recall@1 | 0.250 |
| Recall@3 | 0.333 |
| Recall@5 | 0.417 |

Sensiblemente más bajo que en la Tarea 1 (Recall@5 ≈ 0.75-0.81 allí). Detalle
completo en `eval/results/hybrid_retrieval_detail.csv`.

## Limitación identificada: lenguaje burocrático repetitivo a esta escala

Caso analizado en detalle: la pregunta *"¿Qué proceso es para el
mejoramiento del sistema de agua potable y alcantarillado en San Sebastián de
Choropampa?"* (Cajamarca) recuperó como top-1 un proceso **distinto**, sobre
mejoramiento de agua potable en otra localidad (Shitac, también en Cajamarca),
con mayor similitud (0.772) que el proceso correcto. Ambas descripciones son
casi idénticas en estructura ("CONTRATACIÓN PARA LA EJECUCIÓN DE LA OBRA:
MEJORAMIENTO... DEL SISTEMA DE AGUA POTABLE...") y solo difieren en los nombres
de localidad — nombres de centros poblados pequeños que el modelo multilingüe
no discrimina bien (son poco frecuentes en su entrenamiento).

**Efecto del filtro de departamento:** con `departamento:
Cajamarca` los candidatos bajan de 20,866 a 874, pero el mismo proceso
incorrecto sigue ganando (también es de Cajamarca) — el filtro por departamento
reduce el universo de búsqueda pero no alcanza a discriminar dentro de un
mismo departamento entre docenas de localidades con lenguaje burocrático casi
idéntico. Con corpus de este tamaño y este nivel de repetición textual, ni el
retrieval semántico puro ni el filtro territorial por sí solos garantizan
precisión a nivel de proceso exacto.

## Limitación identificada: el threshold no separa in-domain de out-of-domain

Se evaluaron preguntas claramente fuera de dominio:

| Pregunta | Similitud top-1 |
|---|---|
| "¿Cuál es la capital de Francia?" | 0.371 |
| "¿Cómo se prepara un ceviche peruano?" | **0.696** |
| "¿Cuál es el clima en la luna?" | 0.541 |

**"¿Cómo se prepara un ceviche peruano?" obtiene más similitud (0.696) que
varias preguntas in-domain reales** (ej. h12 con filtros, sim=0.466). A esta
escala, con descripciones cortas y muy repetitivas, el espacio de embeddings
no separa limpiamente preguntas reales de preguntas absurdas — a diferencia de
la Tarea 1, aquí **ningún threshold único logra simultáneamente** 0%
abstención incorrecta sobre las 12 preguntas reales Y abstención correcta
sobre "ceviche". Se optó por priorizar no abstenerse de preguntas reales
(threshold recalibrado a **0.45**, ver `config.yaml`), aceptando que la
defensa principal contra preguntas absurdas recae en el LLM (que debe negarse
a responder citando ocids que no corresponden), no en el threshold de
similitud — una dependencia aún mayor en la "segunda línea de defensa" que en
la Tarea 1.

## Recalibración de threshold

Sweep completo en `eval/results/hybrid_threshold_sweep.csv`. El threshold de
Tarea 1 (0.60) **no transfiere**: a 0.60 ya hay 1/12 (8.3%) de abstención
incorrecta sobre preguntas reales. Se recalibró a **0.45**, el borde superior
de la meseta con 0% de abstención incorrecta sobre las 12 preguntas conocidas.

## Filtros estructurados combinados con búsqueda semántica

La pregunta h12 (*"¿Qué obra hay sobre mejoramiento de un parque central?"*
con `departamento=Piura, categoria=works`) redujo los candidatos de 20,866 a
119 antes de buscar por similitud — confirma que las condiciones
territoriales/numéricas se aplican como filtro de metadata, nunca como texto
embebido (tal como exige el enunciado). El proceso correcto apareció en el
top-5 pero no en el top-1/3, otro caso del mismo problema de boilerplate.
