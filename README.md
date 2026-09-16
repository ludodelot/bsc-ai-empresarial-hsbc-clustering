# 🧠 AI Empresarial — Clustering, Clasificación y LLMs Aplicados

![Featured Project](https://img.shields.io/badge/⭐-Featured%20Project-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI API](https://img.shields.io/badge/OpenAI%20API-GPT--4o--mini%20%7C%20Whisper-412991?style=flat&logo=openai&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20DBSCAN%20%7C%20GMM%20%7C%20RF-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Chat%20UI-FF7C00?style=flat&logo=gradio&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?style=flat&logo=pandas&logoColor=white)

> **El proyecto más avanzado de la carrera.** Cinco notebooks que recorren el ciclo completo de IA aplicada a negocio: segmentación no supervisada de clientes, clasificación supervisada optimizada por costo de negocio, un chatbot conversacional con LLM, un caso de consultoría real con **HSBC** donde GPT-4o-mini genera informes ejecutivos automáticos por cluster B2B, y un pipeline de audio-a-texto con Whisper + NLP. Curados de la materia **AI Empresarial** (S6), Licenciatura en Innovación y Tecnología (**LIT**), Tecnológico de Monterrey.

---

## 📑 Tabla de contenido

- [Mapa del repo](#-mapa-del-repo)
- [Contexto académico](#-contexto-académico)
- [Notebooks incluidos](#-notebooks-incluidos)
- [Deep dive: 01 — Segmentación RFM multi-algoritmo](#-01--segmentación-rfm-multi-algoritmo)
- [Deep dive: 02 — Comparación de clasificadores](#-02--comparación-de-clasificadores)
- [Deep dive: 04 — Caso HSBC: informes ejecutivos con GPT-4o-mini](#-04--caso-hsbc-informes-ejecutivos-con-gpt-4o-mini)
- [Deep dive: 03 — Chatbot LLM con Gradio](#-03--chatbot-llm-con-gradio)
- [Seguridad y manejo de credenciales](#️-seguridad-y-manejo-de-credenciales)
- [Datos](#-datos)
- [Cómo correrlo](#-cómo-correrlo)
- [Estructura del repo](#-estructura-del-repo)
- [Créditos](#-créditos)

---

## 🗺️ Mapa del repo

Cada notebook resuelve un problema de negocio distinto con una técnica de IA distinta. Así se conecta todo:

```mermaid
flowchart LR
    subgraph NEG["🎯 Problema de negocio"]
        N1["Segmentar clientes<br/>por comportamiento de compra"]
        N2["Clasificar clientes<br/>minimizando falsos negativos"]
        N3["Atender consultas<br/>conversacionales"]
        N4["Explicar clusters B2B<br/>en lenguaje ejecutivo"]
        N5["Convertir llamadas de audio<br/>en insights de texto"]
    end

    subgraph NB["📓 Notebook"]
        NB1["01_RFM_Clustering_Clientes"]
        NB2["02_Comparacion_Algoritmos_Clasificacion"]
        NB3["03_Chatbot_LLM_OpenAI_Gradio"]
        NB4["04_HSBC_Clusters_OpenAI_Explain"]
        NB5["05_Audio_a_Texto_Whisper_NLP"]
    end

    subgraph TECH["🛠️ Tecnología"]
        T1["KMeans · DBSCAN · GMM<br/>StandardScaler · t-SNE"]
        T2["RandomForestClassifier<br/>GridSearch · métrica de costo FN"]
        T3["OpenAI Chat Completions<br/>Gradio UI"]
        T4["pandas z-scores<br/>GPT-4o-mini structured output"]
        T5["Whisper API<br/>librosa · LLM sentiment/NER"]
    end

    subgraph OUT["📤 Output"]
        O1["Segmentos de clientes<br/>accionables"]
        O2["Modelo de clasificación<br/>optimizado por negocio"]
        O3["Asistente conversacional<br/>funcional"]
        O4["Informes ejecutivos JSON<br/>por cluster B2B"]
        O5["Transcripciones + análisis<br/>de sentimiento/entidades"]
    end

    N1 --> NB1 --> T1 --> O1
    N2 --> NB2 --> T2 --> O2
    N3 --> NB3 --> T3 --> O3
    N4 --> NB4 --> T4 --> O4
    N5 --> NB5 --> T5 --> O5

    O1 -.insumo para.-> NB4

    style NB4 fill:#412991,color:#fff
    style O4 fill:#412991,color:#fff
```

> 💡 El caso HSBC (`04`) consume conceptualmente la lógica de segmentación explorada en `01`: clusters de clientes → interpretación automática vía LLM. Es el punto donde el pipeline "clásico" de ML y el mundo de los LLMs se encuentran.

---

## 🎓 Contexto académico

| | |
|---|---|
| **Institución** | Tecnológico de Monterrey (Tec de Monterrey), Campus Santa Fe |
| **Programa** | Licenciatura en Innovación y Tecnología (**LIT**) |
| **Materia** | AI Empresarial — CD3002C.601 |
| **Semestre** | S6 |
| **Equipo** | Equipo 5 — Ludovic Delot Bravo, Gonzalo González Méndez, Anakarenina Serrano Ibarra, Mónica Estrada Mondragón |

### Notebooks excluidos de esta selección

Del total de ~24 notebooks de la materia, se excluyeron los que no aportan valor adicional de portafolio o estaban vacíos/rotos: scripts de práctica introductoria (gráficos básicos, procesamiento de datos genérico), el examen parcial completo (contenido mixto de evaluación, no un proyecto autocontenido), notebooks de prompting exploratorio ya cubiertos conceptualmente por el chatbot y el caso HSBC, y un notebook vacío (`20250226_Script_DelotLudovic.ipynb`, 0 bytes).

---

## 📓 Notebooks incluidos

| # | Notebook | Técnica | Tecnologías | Highlight |
|---|---|---|---|---|
| 01 | `RFM_Clustering_Clientes` | Segmentación de clientes (RFM) | KMeans, DBSCAN, GMM, StandardScaler, t-SNE, Silhouette Score | Comparación multi-algoritmo con métricas cuantitativas, no solo un modelo por default |
| 02 | `Comparacion_Algoritmos_Clasificacion` | Clasificación supervisada | RandomForestClassifier, grid de hiperparámetros, ROC-AUC | Métrica de negocio **custom** que penaliza el costo real de falsos negativos, no solo accuracy |
| 03 | `Chatbot_LLM_OpenAI_Gradio` | NLP conversacional | OpenAI Chat Completions, Gradio | 5 personas de chatbot con prompting de rol e inyección de contexto (JSON de transacciones) |
| 04 | `HSBC_Clusters_OpenAI_Explain` | Generación de reportes con LLM | pandas, z-scores por cluster, GPT-4o-mini, structured output | ⭐ Caso real B2B: automatiza informes ejecutivos que normalmente redactaría un analista |
| 05 | `Audio_a_Texto_Whisper_NLP` | Audio-a-texto + NLP | Whisper API, librosa, LLM (sentimiento/NER/resumen) | Pipeline de 3 etapas: transcripción → análisis de sentimiento → extracción de entidades |

---

## 📊 01 — Segmentación RFM multi-algoritmo

Segmentación de clientes por **Recencia, Frecuencia y Monto (RFM)**, comparando tres algoritmos de clustering distintos y eligiendo el mejor con métricas cuantitativas — no a ojo.

```mermaid
flowchart TD
    A["Datos transaccionales crudos<br/>(compras por cliente, fechas, montos)"] --> B["Ingeniería de variables RFM"]
    B --> B1["Recency<br/>días desde última compra"]
    B --> B2["Frequency<br/>número de transacciones"]
    B --> B3["Monetary<br/>gasto total"]
    B1 --> C["Escalado<br/>StandardScaler"]
    B2 --> C
    B3 --> C

    C --> D1["KMeans"]
    C --> D2["DBSCAN"]
    C --> D3["Gaussian Mixture Model"]

    D1 --> E["Evaluación comparativa"]
    D2 --> E
    D3 --> E

    E --> E1["Silhouette Score"]
    E --> E2["Davies-Bouldin Index"]
    E --> E3["Calinski-Harabasz Index"]

    E1 --> F{"Selección del<br/>mejor modelo"}
    E2 --> F
    E3 --> F

    F --> G["Segmentos de clientes<br/>(ej. VIP, en riesgo, nuevos, inactivos)"]

    style F fill:#F7931E,color:#000
    style G fill:#2e7d32,color:#fff
```

**Por qué importa:** en vez de asumir que KMeans es la respuesta correcta, el notebook valida objetivamente cuál algoritmo produce clusters más densos y mejor separados antes de tomar decisiones de negocio sobre ellos. Visualización adicional con t-SNE para inspeccionar la separación de clusters en 2D.

---

## 🌲 02 — Comparación de clasificadores

Búsqueda de hiperparámetros de Random Forest evaluada no solo por accuracy, sino por una **métrica de negocio custom** que penaliza específicamente los falsos negativos (el error más costoso en contextos como riesgo crediticio o churn).

```mermaid
flowchart TD
    A["Dataset etiquetado<br/>(train/test split 70/30)"] --> B["Grid search de<br/>Random Forest<br/>(distintas profundidades)"]
    B --> C["Modelos candidatos<br/>max_depth = 3, 4, 5, ..."]

    C --> D["Evaluación por modelo"]
    D --> D1["Accuracy / Precision / Recall / F1"]
    D --> D2["ROC-AUC"]
    D --> D3["Matriz de confusión"]
    D --> D4["Métrica de negocio custom<br/>(penaliza FN y FP, premia TN)"]

    D1 --> E{"Selección por<br/>métrica de negocio"}
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F["Modelo final<br/>(max_depth óptimo)"]

    style D4 fill:#F7931E,color:#000
    style F fill:#2e7d32,color:#fff
```

**Resultado del equipo:** el modelo con `max_depth=5` fue seleccionado como el mejor — no por tener el mayor accuracy bruto, sino por maximizar la métrica de negocio custom (450 pts) manteniendo precisión (~85.9%), recall (~76.1%) y ROC-AUC (~85.1%) sólidos. Ejemplo real de cómo la elección de modelo cambia cuando se optimiza para el costo real del error, no para una métrica genérica.

---

## 🏦 04 — Caso HSBC: informes ejecutivos con GPT-4o-mini

El notebook más avanzado del repo. Toma clusters de clientes B2B ya calculados (ingresos, EBITDA, saldos normalizados por industria) y usa **GPT-4o-mini** para redactar automáticamente el tipo de informe ejecutivo que normalmente escribiría un analista financiero — perfil del cluster, drivers, causas, recomendaciones y riesgos — en **JSON estructurado**.

```mermaid
sequenceDiagram
    participant Data as Datos B2B normalizados<br/>(z-scores por industria)
    participant Pipe as Pipeline pandas
    participant Cluster as Clusters (impacto_cluster)
    participant LLM as GPT-4o-mini
    participant Report as Informe JSON
    participant Log as Validación / Logging

    Data->>Pipe: cargar métricas financieras
    Pipe->>Pipe: calcular estadísticas agregadas por cluster
    Pipe->>Cluster: agrupar clientes por cluster
    loop Para cada cluster
        Cluster->>LLM: prompt estructurado<br/>("piensa paso a paso" + stats del cluster)
        LLM-->>Report: informe ejecutivo JSON<br/>(perfil, drivers, causas, recomendaciones, riesgos)
        Report->>Log: validar schema del JSON
        Log->>Log: registrar resultado (logging)
    end
    Log-->>Pipe: informes consolidados por cluster
```

**Por qué es el highlight del repo:** conecta un pipeline clásico de analítica de datos (agregaciones, z-scores por industria) con generación de texto estructurado vía LLM, incluyendo manejo de errores, logging y validación de la respuesta — no es solo "un prompt suelto", es un pipeline con `argparse`, `logging` y `dotenv` listo para producción académica.

---

## 💬 03 — Chatbot LLM con Gradio

Chatbot interactivo construido sobre la API de OpenAI con interfaz web funcional en Gradio. Incluye cinco variaciones de personalidad/rol para explorar prompting de sistema, inyección de contexto (JSON de transacciones) y manejo de estado conversacional.

```mermaid
flowchart LR
    U["👤 Usuario"] -->|escribe mensaje| UI["🖥️ Interfaz Gradio"]
    UI -->|prompt + historial| API["OpenAI Chat<br/>Completions API"]
    API -->|respuesta del modelo| UI
    UI -->|muestra respuesta| U

    style API fill:#412991,color:#fff
```

Casos implementados: asistente bancario cortés, asistente con doble comportamiento (rudo salvo en temas de transacciones), asistente con contexto JSON de transacciones de 3 meses, chatbot con "paciencia limitada" y asistente personalizado con datos del usuario.

---

## ⚠️ Seguridad y manejo de credenciales

> [!IMPORTANT]
> Los notebooks originales de clase contenían **API keys de OpenAI reales hardcodeadas en texto plano** — una práctica común, pero insegura, en entregas académicas. Antes de publicar este repositorio se aplicó una limpieza de seguridad explícita:
>
> - Todas las keys hardcodeadas fueron **purgadas del código** y reemplazadas por `os.getenv("OPENAI_API_KEY")`, cargado desde un archivo `.env` vía `python-dotenv`.
> - Se verificó con `grep` exhaustivo que **ningún notebook en `notebooks/` contiene una key real** antes de la publicación.
> - `.env` está en `.gitignore` — **nunca** se commitea un archivo con una key real. Solo `.env.example` (sin valores) vive en el repo.
> - Como práctica recomendada tras cualquier exposición accidental de una key en texto plano, esa key debe revocarse/rotarse en el proveedor (OpenAI) independientemente de que se purgue del código — el código limpio no deshace una exposición ya ocurrida.

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edita .env y coloca tu propia OPENAI_API_KEY
```

Los notebooks que llaman a la API de OpenAI (`03`, `04`, `05`) requieren una key válida en `.env` para ejecutar las celdas que hacen llamadas al modelo. El resto del pipeline (limpieza de datos, clustering, clasificación, visualizaciones) corre **sin necesidad de API key**.

---

## 🗂️ Datos

> [!NOTE]
> **Sobre los datos del caso HSBC:** `data/clientes_clasificados_HSBC.csv` y `data/resumen_clusters_HSBC.csv` corresponden al **material de ejercicio provisto para la actividad académica**. Las columnas son métricas financieras **normalizadas/estandarizadas** (z-scores) por industria (ingresos, utilidad neta, EBITDA, saldos promedio) y una etiqueta de cluster (`impacto_cluster`) — **no contienen nombres, identificadores de cliente, números de cuenta ni ningún dato que permita identificar a una persona o empresa real**. Se incluyen porque son necesarios para reproducir el notebook; en caso de duda, trátense como datos sintéticos/de ejercicio para fines exclusivamente educativos.

Otros archivos en `data/`:

- `rfm_clientes_clusterizados_final.csv` — salida del pipeline de clustering RFM (IDs de cliente numéricos anónimos, sin PII).
- `resultados_rf_opt.csv`, `resultados_rf_afinado.csv` — resultados de la búsqueda de hiperparámetros para el Random Forest de `02_Comparacion_Algoritmos_Clasificacion.ipynb`.

El dataset crudo original de `01_RFM_Clustering_Clientes.ipynb` (`Customer_data-Grid view.csv`) no está disponible en la carpeta fuente al momento de preparar este repo; el notebook fue ajustado para referenciar el CSV de salida ya procesado (`rfm_clientes_clusterizados_final.csv`) como ejemplo reproducible.

---

## ▶️ Cómo correrlo

```bash
git clone <este-repo>
cd bsc-ai-empresarial-hsbc-clustering
pip install -r requirements.txt
cp .env.example .env
# Edita .env con tu OPENAI_API_KEY
jupyter notebook notebooks/
```

- `01` y `02` corren de punta a punta sin API key.
- `03`, `04` y `05` necesitan `OPENAI_API_KEY` configurada en `.env` para las celdas que llaman a la API de OpenAI (Chat Completions y Whisper).

---

## 📁 Estructura del repo

```
bsc-ai-empresarial-hsbc-clustering/
├── notebooks/
│   ├── 01_RFM_Clustering_Clientes.ipynb
│   ├── 02_Comparacion_Algoritmos_Clasificacion.ipynb
│   ├── 03_Chatbot_LLM_OpenAI_Gradio.ipynb
│   ├── 04_HSBC_Clusters_OpenAI_Explain.ipynb
│   └── 05_Audio_a_Texto_Whisper_NLP.ipynb
├── data/
│   ├── clientes_clasificados_HSBC.csv
│   ├── resumen_clusters_HSBC.csv
│   ├── rfm_clientes_clusterizados_final.csv
│   ├── resultados_rf_opt.csv
│   └── resultados_rf_afinado.csv
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🙌 Créditos

**Equipo 5** — Materia AI Empresarial, S6, Licenciatura en Innovación y Tecnología (LIT), Tecnológico de Monterrey:

- Ludovic Delot Bravo (A01663977)
- Gonzalo González Méndez (A01784359)
- Anakarenina Serrano Ibarra (A01783752)
- Mónica Estrada Mondragón (A017716269)

Profesora titular: Fabiola Celia Vásquez García.

---

Autor de esta curaduría de portafolio: **Ludovic Delot Bravo**
