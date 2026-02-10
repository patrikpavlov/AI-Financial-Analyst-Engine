# AI Financial Analyst Engine

A small Python project that generates concise, analyst-style summaries for **US stocks** using:
- a **RAG pipeline** (`rag_engine.py`)
- an app entry point (`app.py`)
- an optional **fine-tuning script** for Llama-3 (`llama_3_8b_instruct_finetune.py`)

> Educational/demo project — not financial advice.

## Quickstart
```bash
git clone https://github.com/patrikpavlov/AI-Financial-Analyst-Engine.git
cd AI-Financial-Analyst-Engine

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
