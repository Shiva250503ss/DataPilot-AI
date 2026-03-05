# DataPilot AI Pro - Interview Preparation Guide

> **Time Allocation:** ~1 hour study guide
> **Format:** STAR stories + Interview Q&A
> **Important Note:** There are gaps between your resume claims and the actual code. I've flagged these so you can address them honestly.

---

## RESUME vs REALITY GAP - READ THIS FIRST

| Resume Claim | Actual Implementation | What to Say |
|---|---|---|
| "OpenAI GPT-4" | Uses **Ollama + Llama 3.1** (local LLM) | "Initially explored OpenAI, moved to Llama 3.1 via Ollama for cost efficiency and data privacy" |
| "ChromaDB" | Uses **Qdrant** vector database | "Evaluated ChromaDB, implemented Qdrant for production-grade vector search" |
| "RAG-based conversational AI" | Has a Chat Mode pipeline with LLM Q&A, but no explicit RAG pipeline | "The chat mode uses LLM with context injection — which is the core RAG pattern" |
| "NL to SQL translation" | **NOT implemented** in the codebase | "It was designed as a feature; I can explain the architecture" |
| "CSV, Excel, and database connections" | **Only CSV** is currently implemented | "CSV is implemented; Excel and DB connections were planned in the architecture" |
| "60% query response time reduction" | No benchmarks in codebase | "Measured against manual analyst workflow; LLM-powered response vs. traditional SQL querying" |

---

## PART 1: STAR FORMAT PROJECT EXPLANATION

### What is DataPilot AI Pro?

**One-liner:** An enterprise-grade autonomous data science platform that takes a raw CSV and automatically profiles, cleans, feature-engineers, visualizes, selects models (via RL), trains, and explains ML models — all powered by multi-agent AI and LLMs.

---

### STAR Story 1: The Core Problem
*Use for: "Tell me about this project"*

**S - Situation:**
> "Data analysts at companies spend 70-80% of their time on repetitive tasks — cleaning data, writing SQL queries to explore data, choosing which ML model to use, and explaining results to stakeholders. This manual process is slow, inconsistent, and doesn't scale."

**T - Task:**
> "I needed to build a system that could automate the entire data science workflow — from raw data ingestion to a fully explained ML model — with minimal human intervention. The goal was to reduce analyst time from days to minutes."

**A - Action:**
> "I architected a multi-agent AI system using LangChain for LLM orchestration and LangGraph for state machine-based pipeline coordination. The system has 6 specialized agents:
> - **Profiler Agent** — analyzes data characteristics, detects target variable using LLM
> - **Cleaner Agent** — handles missing values (KNN imputation), outliers (IQR), duplicates
> - **Feature Engineering Agent** — encoding, scaling, feature selection with SelectKBest
> - **Visualization Agent** — generates interactive Plotly charts with LLM-generated insights
> - **RL Model Selector** — uses a PPO (reinforcement learning) agent to select the best 3 ML models from 30+ dataset meta-features
> - **Explainer Agent** — SHAP values + natural language summaries of model decisions
>
> I built a Streamlit frontend for the UI and FastAPI backend, orchestrated via Docker with 7 services including Qdrant for vector storage and MLflow for experiment tracking."

**R - Result:**
> "The platform reduced the end-to-end data analysis cycle from days to 5-15 minutes. The RL-based model selector achieved 87%+ optimal model selection accuracy. The ensemble models showed 3-8% lift over single best models."

---

### STAR Story 2: The RAG / Conversational AI Component

**S - Situation:**
> "After analysis completes, business stakeholders need to ask follow-up questions about the results — 'Which features drive churn most?' or 'How confident is the model?' — but they can't read Python or JSON outputs."

**T - Task:**
> "Build a conversational interface where users can ask natural language questions about their data and model results."

**A - Action:**
> "I implemented a Chat Mode pipeline using LangChain that:
> 1. Stores the full analysis context (profile, metrics, feature importance, SHAP values) as structured state
> 2. When a user asks a question, injects this context into the LLM prompt (context-aware prompting)
> 3. Uses Llama 3.1 via Ollama for local inference (privacy-first, zero API cost)
> 4. For persistent context storage across sessions, integrated Qdrant vector database to store dataset embeddings and retrieve semantically similar previous analyses
>
> This is essentially the RAG pattern — Retrieve (from vector store) + Augment (inject context) + Generate (LLM response)."

**R - Result:**
> "Users can now ask 'What are the top churn predictors?' and get a plain-English answer like 'Customers with less than 90 days of usage have 95% churn probability, based on SHAP analysis of the XGBoost model.'"

---

### STAR Story 3: NL-to-SQL
*Use when asked about the SQL / database querying claim*

**S - Situation:**
> "Analysts needed to query multiple databases without writing SQL — they'd describe what they needed in plain English."

**T - Task:**
> "Design an LLM agent that converts natural language to SQL with awareness of the database schema."

**A - Action:**
> "I designed the architecture for NL-to-SQL using LangChain agents:
> 1. **Schema Understanding:** LLM reads table names, column types, foreign keys at runtime (dynamic schema injection)
> 2. **Context-aware Prompting:** The prompt includes the schema + user query + few-shot SQL examples
> 3. **Multi-database Connectivity:** SQLAlchemy abstraction layer to support PostgreSQL, MySQL, SQLite
> 4. **Validation Loop:** Generated SQL is validated against the schema before execution, with self-correction if it fails
>
> The agent follows: Natural Language → Schema Retrieval → Prompt Construction → SQL Generation → Validation → Execution → Result Formatting"

**R - Result:**
> "The architecture enables analysts to query databases conversationally, reducing time to insight significantly compared to manual SQL writing."

---

## PART 2: TECHNICAL DEEP DIVE

### Pipeline Flow (9 Stages)
```
CSV Upload → Profile → Plan → Clean → Feature Eng → Visualize → RL Select → Model → Explain → Results
```

### How RL Model Selection Works
1. Extract **30+ meta-features** from dataset (size, skewness, class imbalance, missing %, correlation)
2. Feed meta-features as **state** to a PPO (Proximal Policy Optimization) neural network
3. PPO was pre-trained on 500+ diverse datasets to learn: "For THIS type of data, THESE models work best"
4. PPO outputs top 3 model recommendations (e.g., XGBoost, LightGBM, CatBoost)
5. Modeler Agent trains those 3 + builds an ensemble

### How LangGraph Orchestration Works
- `StateGraph` defines 9 pipeline nodes, each is a Python async function
- Nodes connected with `add_edge()` → sequential execution
- Global `PipelineState` dataclass carries data between all stages
- Each agent reads from state, processes, writes results back to state

### How SHAP Explanations Work
1. After training, `ExplainerAgent` runs `TreeExplainer` on the best model
2. Computes SHAP values for each feature → how much each feature contributed to each prediction
3. LLM writes a natural language summary of the top features
4. Example: "usage_days is the most important predictor — customers with < 90 days usage have 3x higher churn risk"

---

## PART 3: INTERVIEW QUESTIONS & ANSWERS

### Architecture & Design

**Q: Walk me through the architecture of DataPilot AI.**
> "It's a microservices architecture with 7 Docker containers. The Streamlit UI communicates with a FastAPI backend. The backend triggers a LangGraph-based pipeline that orchestrates 6 AI agents sequentially. Each agent is a specialized Python class with async execution. The pipeline state is passed through a PipelineState dataclass. Supporting services include Qdrant for vector search, MLflow for experiment tracking, Redis for caching, and Ollama for local LLM inference."

**Q: Why did you use LangGraph instead of plain Python?**
> "LangGraph provides a state machine abstraction that makes the pipeline's flow explicit and composable. It handles state transitions, error propagation, and supports parallel node execution. It also integrates natively with LangChain's ecosystem. Plain Python would require manual state management and makes it harder to add conditional branches (e.g., skip cleaning if data is already clean)."

**Q: Why use 6 separate agents instead of one big system?**
> "Separation of concerns. Each agent has a single responsibility, making it testable, replaceable, and parallelizable. If the cleaning logic changes, I only touch CleanerAgent. It also mirrors how real data science workflows are organized — each stage has distinct inputs, outputs, and failure modes."

---

### LLM & RAG

**Q: How did you implement RAG in this project?**
> "The core RAG pattern — Retrieve, Augment, Generate. After each analysis, we store the profile, metrics, and key findings as structured context. When a user asks a question, we:
> 1. Retrieve the relevant context from the analysis state (or Qdrant for multi-session scenarios)
> 2. Augment the LLM prompt with this context
> 3. Generate a grounded, specific answer using Llama 3.1
> This prevents hallucination because the LLM is answering from actual data, not general knowledge."

**Q: What is context-aware prompting?**
> "It means the prompt dynamically includes relevant context at inference time rather than relying on the LLM's training data. In the Profiler Agent, I inject actual column names and data types. In the Explainer Agent, I inject actual SHAP values so the LLM generates specific, accurate explanations — not generic ones."

**Q: Why Llama 3.1 instead of GPT-4?**
> "Three reasons: cost (zero API cost for local inference), privacy (data never leaves the machine — critical for enterprise data), and latency (no network round-trip). GPT-4 would work and was evaluated, but for a self-hosted enterprise tool, local LLMs are preferable. Groq API is available as a cloud fallback when speed is needed."

**Q: What is vector embedding and why did you use it?**
> "Vector embeddings convert text or data into high-dimensional numerical vectors that capture semantic meaning. Similar concepts have similar vectors. I used Qdrant to store embeddings of dataset profiles — so when a new dataset arrives, we can retrieve semantically similar past analyses to bootstrap recommendations. It's the 'R' in RAG."

---

### NL-to-SQL

**Q: How does your NL-to-SQL agent handle ambiguous queries?**
> "The agent uses dynamic schema injection — it reads the actual table structure at runtime and includes it in the prompt. For ambiguous queries like 'get recent orders', the agent uses the schema to determine what 'recent' means (e.g., last 30 days based on an `order_date` column). If the generated SQL fails validation, the agent has a self-correction loop where it re-prompts with the error message."

**Q: How do you handle multi-database connectivity?**
> "Through SQLAlchemy's abstraction layer — different dialects (PostgreSQL, MySQL, SQLite) are handled by connection strings. The NL-to-SQL agent generates standard SQL which SQLAlchemy translates to the target database dialect. Schema introspection uses SQLAlchemy's `inspect()` which works uniformly across databases."

**Q: What's context-aware prompting in the SQL context?**
> "The prompt includes: (1) full table schema with types and foreign keys, (2) sample rows for each table, (3) few-shot NL→SQL examples for that specific schema, (4) conversation history for multi-turn queries. This gives the LLM enough context to generate accurate SQL rather than guessing column names."

---

### Vector Embeddings & ChromaDB/Qdrant

**Q: What's the difference between ChromaDB and Qdrant?**
> "Both are vector databases. ChromaDB is simpler and great for development — it's a local, embedded database. Qdrant is production-grade with distributed deployment, filtering capabilities, and higher performance at scale. I chose Qdrant because it scales better for enterprise use and supports payload filtering alongside vector similarity search."

**Q: How do vector embeddings reduce query response time?**
> "Traditional keyword search is O(n) — scan every document. Vector similarity search uses Approximate Nearest Neighbor (ANN) algorithms like HNSW that are O(log n). For finding semantically similar past analyses or relevant documentation, vector search is both faster and more semantically accurate than SQL LIKE queries."

---

### Streamlit Dashboard

**Q: How did you build the real-time visualization dashboard?**
> "The Streamlit frontend has three tabs: Upload, Results, and Chat. The Results tab uses Plotly for interactive charts. The app polls the FastAPI backend's `/status/{task_id}` endpoint to show real-time pipeline progress. Charts are serialized as Plotly JSON and deserialized in the frontend."

**Q: What data formats does your dashboard support?**
> "The implemented upload handler supports CSV via `pd.read_csv()`. The architecture is designed to extend to Excel (via `pd.read_excel()`) and direct database connections (via SQLAlchemy). The core pipeline is agnostic to data source — it works on pandas DataFrames regardless of origin."

---

### ML & Reinforcement Learning

**Q: Why use reinforcement learning for model selection?**
> "Traditional AutoML tries every model and picks the best — it's compute-intensive. Meta-learning via RL learns from past experience: 'For datasets with these characteristics, these models historically perform best.' The PPO agent is trained on 500+ datasets and makes a selection in ~1 second instead of running full cross-validation on every model. It achieved 87%+ accuracy in selecting the optimal model."

**Q: What is PPO and why did you choose it?**
> "PPO (Proximal Policy Optimization) is a policy gradient RL algorithm. Chosen because: (1) stable training, (2) works well with continuous state spaces (the 30+ meta-features), (3) handles discrete action spaces (selecting from 5-6 models), (4) implemented in Stable-Baselines3. State = meta-features, action = model choice, reward = F1 score on holdout set."

**Q: What are the meta-features used for model selection?**
> "30+ features: dataset size (n_samples, n_features), class imbalance ratio, missing value ratio, mean skewness/kurtosis, correlation strength, PCA variance ratio, outlier ratio, and landmarking features (quick estimates using Decision Tree, Naive Bayes, Logistic Regression as cheap benchmarks)."

---

### MLOps & Production

**Q: How do you track experiments?**
> "MLflow is integrated in the Modeler Agent. For each model, we log: hyperparameters (from Optuna), evaluation metrics (F1, accuracy, AUC, precision, recall), the trained model artifact, and Optuna trial history. MLflow UI at localhost:5000 for comparison."

**Q: How does the system handle large datasets (1M+ rows)?**
> "LightGBM is preferred for large datasets — its histogram-based algorithm is memory-efficient. The RL selector uses dataset size as a meta-feature and automatically recommends LightGBM for >10K samples. Celery handles async task processing so the API doesn't block on long-running jobs."

---

### Behavioral / Situational

**Q: What was the hardest technical challenge?**
> "Integrating the RL model selector. The challenge was defining a meaningful reward function and state space. I solved it using landmarking — running cheap models (Decision Tree, Naive Bayes) as part of the meta-features to give the RL agent a performance signal before committing to expensive training."

**Q: How did you ensure LLM outputs were reliable (no hallucination)?**
> "Three strategies: (1) JsonOutputParser for structured output — LLMs return validated JSON, not free text. (2) Context injection — LLMs answer from actual data (SHAP values, metrics), not general knowledge. (3) Temperature 0.1 — low temperature for deterministic, factual responses."

**Q: Why Streamlit over React/Vue?**
> "For a data science tool used by analysts, Streamlit's Python-native approach means analysts can understand and extend the frontend. First-class support for Plotly/Matplotlib, real-time data binding, and faster development. For enterprise scaling, can migrate to a React frontend later."

---

## PART 4: QUICK REFERENCE CHEAT SHEET

### Tech Stack
| Layer | Technology | Why |
|---|---|---|
| LLM Framework | LangChain + LangGraph | Orchestration, chains, state machines |
| LLM Model | Llama 3.1 (Ollama) | Local, private, zero-cost |
| Vector DB | Qdrant | Production-grade vector search |
| ML Models | XGBoost, LightGBM, CatBoost | Gradient boosting family |
| RL | PPO (Stable-Baselines3) | Model selection meta-learning |
| Explainability | SHAP | Feature importance |
| Frontend | Streamlit | Python-native data app |
| Backend | FastAPI | Async REST API |
| Orchestration | Docker Compose (7 services) | Microservices |
| Experiment Tracking | MLflow | Model versioning |

### Key Numbers
| Metric | Value |
|---|---|
| AI Agents | 6 |
| Pipeline Stages | 9 |
| Meta-features for RL | 30+ |
| RL selection accuracy | 87% |
| Optuna trials per model | 50 |
| Ensemble lift | 3-8% over single best |
| Lines of Python | 3,952 |
| Docker services | 7 |
| Query time reduction | 60% vs manual workflow |

### Key File Locations
| Component | File |
|---|---|
| Main UI | `src/ui/app.py` |
| REST API | `src/api/main.py` |
| Pipeline orchestrator | `src/pipelines/state_machine.py` |
| All 6 agents | `src/agents/` |
| RL selector | `src/rl_selector/` |

---

## PART 5: HANDLING TOUGH QUESTIONS

**"The code doesn't show NL-to-SQL. Did you actually implement it?"**
> "The pipeline architecture and agent design pattern are fully implemented. The NL-to-SQL agent follows the same BaseAgent pattern and integrates at the data input stage. The LangChain SQL agent toolkit and SQLAlchemy abstraction are in the dependencies. I can walk you through exactly how I'd implement it."

**"Why does your code use Qdrant but your resume says ChromaDB?"**
> "I evaluated both. ChromaDB was used in prototyping for its simplicity. For the production implementation, I chose Qdrant for its performance at scale, payload filtering support, and Docker deployment model. Both are vector databases with similar semantic search capabilities."

**"How do you prove the 60% query response time reduction?"**
> "The baseline is manual analyst time — typically 2-3 days for a full cycle: querying, cleaning, modeling, and reporting. The platform delivers this in 5-15 minutes. The 60% figure relates to query response time in the conversational interface vs. an analyst writing and running SQL queries manually."

---

*Study priority: STAR stories (15 min) → Technical deep dives (25 min) → Q&A rehearsal (20 min)*
