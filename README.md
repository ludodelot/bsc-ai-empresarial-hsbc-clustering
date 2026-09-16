# AI Empresarial — Clustering, Clasificación y LLMs Aplicados

Selección curada de proyectos de la materia **AI Empresarial** (Semestre 6, "S6P1_AI_EMPRESARIAL"), Licenciatura en Inteligencia de Negocios, **Tecnológico de Monterrey (Tec de Monterrey)**.

Este repositorio reúne el trabajo más avanzado de la carrera en ciencia de datos aplicada y modelos de lenguaje (LLMs), incluyendo un caso de negocio real trabajado con **HSBC**: generación de informes ejecutivos estructurados (JSON) por cluster de clientes B2B usando GPT-4o-mini.

## Contenido

| Notebook | Descripción | Técnicas |
|---|---|---|
| `01_RFM_Clustering_Clientes.ipynb` | Segmentación de clientes por recencia, frecuencia y monto (RFM), comparando múltiples algoritmos de clustering. | KMeans, DBSCAN, Gaussian Mixture Models, StandardScaler, t-SNE, silhouette score |
| `02_Comparacion_Algoritmos_Clasificacion.ipynb` | Comparación sistemática de algoritmos de clasificación con Random Forest afinado por hiperparámetros, evaluado con lógica de negocio explícita sobre el costo de falsos negativos. | RandomForestClassifier, grid de hiperparámetros, accuracy/precision/recall/F1/ROC-AUC, matriz de confusión |
| `03_Chatbot_LLM_OpenAI_Gradio.ipynb` | Chatbot interactivo funcional construido sobre la API de OpenAI con interfaz web en Gradio. | OpenAI Chat Completions, Gradio |
| `04_HSBC_Clusters_OpenAI_Explain.ipynb` | **Caso HSBC.** Pipeline que calcula estadísticas por cluster de clientes B2B (ingresos, EBITDA, saldos) y usa GPT-4o-mini para generar automáticamente informes ejecutivos estructurados (perfil, drivers, causas, recomendaciones, riesgos) en formato JSON. | pandas, z-scores por cluster, prompting estructurado "piensa paso a paso", `gpt-4o-mini` |
| `05_Audio_a_Texto_Whisper_NLP.ipynb` | Transcripción de audio a texto con Whisper y análisis NLP posterior (sentimiento, extracción de entidades, resumen) sobre el texto transcrito. | OpenAI Whisper API, librosa, análisis de sentimiento con LLM |

### Notebooks excluidos de esta selección

Del total de ~24 notebooks de la materia, se excluyeron los que no aportan valor adicional de portafolio o estaban vacíos/rotos: scripts de práctica introductoria (gráficos básicos, procesamiento de datos genérico), el examen parcial completo (contenido mixto de evaluación, no un proyecto autocontenido), notebooks de prompting exploratorio ya cubiertos conceptualmente por el chatbot y el caso HSBC, y un notebook vacío (`20250226_Script_DelotLudovic.ipynb`, 0 bytes).

## ⚠️ Seguridad: manejo de credenciales

Los notebooks originales de clase contenían **API keys de OpenAI reales hardcodeadas en texto plano** (una práctica común, pero insegura, en entregas académicas). Antes de publicar este repositorio:

- Todas las keys hardcodeadas fueron **purgadas del código** y reemplazadas por `os.getenv("OPENAI_API_KEY")`, cargado desde un archivo `.env` (vía `python-dotenv`).
- Se verificó con `grep` exhaustivo que **ningún notebook en `notebooks/` contiene una key real** antes de la publicación.
- Las keys originalmente expuestas ya fueron rotadas/invalidadas por su dueño; de cualquier forma, nunca debes commitear un archivo `.env` con una key real (ver `.gitignore`).

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edita .env y coloca tu propia OPENAI_API_KEY
```

Los notebooks que llaman a la API de OpenAI (03, 04, 05) requieren una key válida en `.env` para ejecutar las celdas que hacen llamadas al modelo; el resto del pipeline (limpieza de datos, clustering, clasificación, visualizaciones) corre sin necesidad de API key.

## Nota sobre los datos del caso HSBC

Los archivos `data/clientes_clasificados_HSBC.csv` y `data/resumen_clusters_HSBC.csv` corresponden al **material de ejercicio provisto para la actividad académica** del caso HSBC. Las columnas son métricas financieras **normalizadas/estandarizadas** (z-scores) por industria (ingresos, utilidad neta, EBITDA, saldos promedio) y una etiqueta de cluster (`impacto_cluster`) — **no contienen nombres, identificadores de cliente, números de cuenta ni ningún dato que permita identificar a una persona o empresa real**. Se incluyen porque son necesarios para reproducir el notebook y no presentan indicios de ser datos reales de clientes bancarios sin anonimizar; en caso de duda, trátense como datos sintéticos/de ejercicio para fines exclusivamente educativos.

## Datos incluidos en `data/`

- `clientes_clasificados_HSBC.csv`, `resumen_clusters_HSBC.csv` — material del caso HSBC (ver nota arriba).
- `rfm_clientes_clusterizados_final.csv` — salida del pipeline de clustering RFM (IDs de cliente numéricos anónimos, sin PII).
- `resultados_rf_opt.csv`, `resultados_rf_afinado.csv` — resultados de la búsqueda de hiperparámetros para el Random Forest de `02_Comparacion_Algoritmos_Clasificacion.ipynb`.

El dataset crudo original de `01_RFM_Clustering_Clientes.ipynb` (`Customer_data-Grid view.csv`) no está disponible en la carpeta fuente al momento de preparar este repo; el notebook fue ajustado para referenciar el CSV de salida ya procesado (`rfm_clientes_clusterizados_final.csv`) como ejemplo reproducible.

## Contexto académico

- **Institución:** Tecnológico de Monterrey (Tec de Monterrey)
- **Programa:** Licenciatura en Inteligencia de Negocios (LIN)
- **Materia:** AI Empresarial
- **Semestre:** S6

---

Autor: Ludovic Delot Bravo
