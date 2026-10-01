# Clinical Data Fusion & Adaptive Optimizer Agents (LangGraph)
## 6-Agent CDSS Architecture with 4-Input Dual-Confidence Scoring & Adaptive Feedback

This repository implements the **Clinical Data Fusion Agent** and **Adaptive Optimizer Agent** for an advanced 6-agent Clinical Decision Support System (CDSS), extending the IEEE Access research paper:
> *Çağatay Umut Öğdü, Kübra Arslanoğlu, and Mehmet Karaköse*, **"An Adaptive Multi-Agent LLM-Based Clinical Decision Support System Integrating Biomedical RAG and Web Intelligence"**, *IEEE Access*, 2025.

---

## 1. System Overview: From 5 Agents to 6 Agents

The base paper defines a 5-agent pipeline. We extended the architecture by adding a **Graph RAG Agent** for structured knowledge graph traversal (disease-symptom-drug relational triples, comorbidities, contraindications).

The 6 agents are:
1. **Clinical Text Clarifier Agent**: Preprocesses and normalizes unstructured clinical patient text.
2. **Symptom RAG Analyzer Agent (Vector Agent)**: Performs semantic BioBERT vector retrieval on biomedical literature and textbook passages.
3. **Graph RAG Agent (Graph Agent)**: Queries structured biomedical knowledge graphs (e.g. SNOMED CT / ICD / UMLS relations) for comorbidities, drug-drug interactions, and clinical contraindications.
4. **Evidence-Based Scanner Agent (Web Evidence Agent)**: Performs targeted literature searches (PubMed, Google Scholar, clinical guidelines like NICE/IDSA) for both the vector candidate diagnoses and the graph relations.
5. **Clinical Data Fusion Agent (Our Agent #1)**: Integrates and cross-checks the 4 input evidence streams into a unified clinical report.
6. **Optimizer Agent (Our Agent #2)**: Reports the confidence scores and whether C1 meets the threshold ($\tau = 0.65$). The standalone pipeline runs once.

```
                      +-----------------------------+
                      | Clinical Text Clarifier     |
                      +--------------+--------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
      +----------------------------+   +----------------------------+
      | 1. Vector RAG Agent        |   | 3. Graph RAG Agent         |
      +-------------+--------------+   +--------------+-------------+
                    |                                 |
                    v                                 v
      +----------------------------+   +----------------------------+
      | 2. Evidence Agent on Vector|   | 4. Evidence Agent on Graph |
      +-------------+--------------+   +--------------+-------------+
                    |                                 |
                    +----------------+----------------+
                                     | (4 Inputs)
                                     v
                      +-----------------------------+
                      | Clinical Data Fusion Agent  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Dual Confidence Calculation |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Adaptive Optimizer Agent   |
                      +--------------+--------------+
                                     |
                                     | (standalone run ends here)
                                     v
                                   [END]
```

---

## 2. The 4 Inputs to the Fusion Agent

In this revised architecture, the Fusion Agent receives **4 distinct inputs**:

| # | Input Name | Source Agent | Description |
|---|---|---|---|
| **1** | `vector_evidence` | **Vector Agent** | BioBERT cosine similarity retrieval, candidate preliminary diagnoses, and clinical findings. |
| **2** | `vector_web_evidence` | **Evidence Agent on Vector** | Clinical guidelines, meta-analyses, and literature validating the Vector Agent's proposed diagnoses. |
| **3** | `graph_evidence` | **Graph Agent** | Knowledge graph entities, relationship triples, drug-drug interactions, and patient comorbidities. |
| **4** | `graph_web_evidence` | **Evidence Agent on Graph** | Literature, FDA safety alerts, and systematic reviews validating the Graph Agent's interaction and safety constraints. |

---

## 3. Dual-Track Confidence Scoring & Mathematical Formulation

Rather than a single combined confidence score, the system computes **two independent confidence scores** grounded in objective clinical signals, then fuses them:

### A. Confidence Score 1 ($C_1$ - Vector Stream)
Evaluates the reliability of the vector diagnosis validated against its retrieved literature:
$$C_1 = w_{\text{sim}} \cdot S_{\text{vec}} + w_{\text{web}} \cdot E_{\text{web\_vec}} + w_{\text{conc}} \cdot \text{Conc}_{\text{vec}}$$
- $S_{\text{vec}} \in [0, 1]$: BioBERT cosine similarity score (e.g. $0.88$).
- $E_{\text{web\_vec}} \in [0, 1]$: Quality score from the Clinical Evidence Hierarchy (Guidelines/Systematic Reviews $= 0.90$, Clinical Trials $= 0.70$, Case Reports $= 0.50$, General $= 0.60$).
- $\text{Conc}_{\text{vec}} \in [0, 1]$: Semantic concordance between the vector diagnosis and retrieved literature.
- **Default Weights**: $w_{\text{sim}} = 0.45$, $w_{\text{web}} = 0.35$, $w_{\text{conc}} = 0.20$ ($\sum w = 1.0$).

### B. Confidence Score 2 ($C_2$ - Graph Stream)
Evaluates the reliability of the Knowledge Graph findings validated against safety literature:
$$C_2 = w_{\text{graph}} \cdot S_{\text{graph}} + w_{\text{web}} \cdot E_{\text{web\_graph}} + w_{\text{conc}} \cdot \text{Conc}_{\text{graph}}$$
- $S_{\text{graph}} \in [0, 1]$: Knowledge graph structural fidelity score ($0.90$ baseline for curated ontologies like SNOMED CT/UMLS).
- $E_{\text{web\_graph}} \in [0, 1]$: Literature evidence quality for drug interactions and contraindications.
- $\text{Conc}_{\text{graph}} \in [0, 1]$: Semantic concordance between graph relation triples and safety literature.
- **Default Weights**: $w_{\text{graph}} = 0.50$, $w_{\text{web}} = 0.30$, $w_{\text{conc}} = 0.20$ ($\sum w = 1.0$).

### C. Recommended Fusion Formula: Dual-Stream Evidential Fusion
To fuse $C_1$ and $C_2$ into a final confidence score ($C_{\text{final}}$):
$$C_{\text{final}} = \text{clamp}\Big(\beta_{\text{vec}} \cdot C_1 + \beta_{\text{graph}} \cdot C_2 + \Delta_{\text{agreement}},\, 0.0,\, 1.0\Big)$$
- $\beta_{\text{vec}} = 0.45$: Weight for vector evidence stream.
- $\beta_{\text{graph}} = 0.55$: Weight for knowledge graph stream (reflects high factual certainty of structured medical knowledge graphs).
- $\Delta_{\text{agreement}} \in \{-0.05, 0.00, +0.05\}$: Cross-stream concordance factor:
  - $+0.05$ if the graph and vector streams mutually validate each other without conflicts.
  - $-0.05$ if an unmitigated clinical contraindication or drug-disease conflict is detected.
  - $0.00$ if independent or neutral.

---

## 4. Threshold Status in the Optimizer Agent

The Optimizer reports threshold status after one pass:
- **Condition Evaluated**: **Confidence Score 1 ($C_1$) ONLY**.
- As per project requirements:
  - If $C_1 \ge \tau$ ($\tau = 0.65$): The Optimizer reports that the score meets the threshold.
  - If $C_1 < \tau$: The Optimizer reports that the score is below the threshold.
  - **Confidence Score 2 ($C_2$) is informational** and is still included in the fused score.

---

## 5. File Structure and Roles

```
.
├── evidence_adapter.py    # Adapter layer: converts ANY shape (str, dict, list) from upstream into clean text
├── confidence_scoring.py  # Pure math: computes C1, C2, and dual-stream evidential fusion
├── fusion_agents.py       # LLM agents: Fusion Agent synthesis and Optimizer Agent feedback & fallbacks
├── graph.py               # LangGraph wiring: 2 nodes (fusion + optimizer), state definition
├── run.py                 # Main single-pass execution script
├── test_pipeline.py       # Unit test suite covering adapter, confidence scoring, and graph
├── requirements.txt       # Dependencies with pinned versions
├── example_input.json     # Example 4-stream input for the pipeline
├── example_output.json    # Example pipeline output (confidence + report + optimizer response)
└── .gitignore
```

---

## 6. How to Run

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Configure API Key:
Create a `.env` file in the project root and add your OpenRouter API key and optional model:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=your-preferred-openrouter-model
```
`OPENROUTER_API_KEY` is required for LLM-generated reports. `OPENROUTER_MODEL` is optional and defaults to `gpt-4o-mini`; set it to a model available to your key. If the API key is missing or an LLM call fails, startup reports the missing key and the pipeline uses its fallback. The output includes a `fallback_used` boolean.

The KG wrapper may pass `kg_safety_messages` as a list of objects with `type`, `drug`, `target`, `severity`, and `message`. Fusion includes these items with their supplied severity and message in the report.

### Run the Standard Pipeline:
```bash
python run.py
```

### Run with Low-Confidence Sample Data:
This single pass displays the optimizer's threshold status.
```bash
python run.py --low-confidence
```

### Run the Unit Test Suite:
```bash
python test_pipeline.py
```

`safety.py` was removed because it was not connected to the pipeline and inferred treatment and safety statements. Safety findings should come from the actual evidence inputs.
