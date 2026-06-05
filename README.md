# Homework 2 — Multi-Agent AutoML Team

Autonomous three-agent pipeline: **Data Cleaner → Feature Engineer → Model Trainer**, with LLM-driven tool use and an XGBoost training feedback loop.

## Setup

```bash
cd Homework2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY
```

## Run

```bash
python scripts/prepare_sample_data.py   # Titanic CSV → data/raw_data.csv
python main.py
```

## Deliverables (generated in `output/`)

| File | Description |
|------|-------------|
| `clean_data.csv` | Agent 1 handoff |
| `engineered_data.csv` | Agent 2 handoff |
| `agent1_structured_report.json` | Structured JSON report from Agent 1 |
| `agent2_structured_report.json` | Structured JSON report from Agent 2 |
| `agent3_structured_report.json` | Structured JSON report from Agent 3 |
| `agent_logs.txt` | Timestamped agent thoughts and tool calls |
| `FINAL_REPORT.md` | End-to-end summary and metrics |

## Architecture

```
raw_data.csv
    → Agent 1 (tools: inspect, impute, outliers, drop, …) → clean_data.csv + structured report
    → Agent 2 (tools: interactions, encode, select, …) → engineered_data.csv + structured report
    → Agent 3 (execute_python_code + feedback loop) → XGBoost metrics + summary
    → FINAL_REPORT.md
```

Decisions are made by the LLM via tool calls — no hardcoded cleaning or hyperparameter defaults in the orchestrator.

## Custom dataset

Replace `data/raw_data.csv` with your CSV (include a clear target column name), then run `python main.py`.

## Checklist

- [x] Agent 1 cleans and passes data to Agent 2
- [x] Agent 2 can create new features via `create_interaction`
- [x] Agent 3 generates and executes Python training code
- [x] Agent 3 feedback loop on accuracy/recall/F1
