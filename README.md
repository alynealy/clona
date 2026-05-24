# CheckWise

<<<<<<< HEAD
Multi-agent framework for high-precision content forensics. Uses a hybrid architecture of cloud-based web grounding and local statistical analysis to detect machine-generated patterns, rhythmic monotony, and factual hallucinations.
=======
CheckWise is the existing Vite/React UI for AI-generated text detection.

## Statistical Agent

The repository now includes a Python statistical agent in [`checkwise_stats/`](./checkwise_stats) and a minimal backend API in [`backend/`](./backend). The intended application path is now:

1. frontend sends a statistical question and dataset to `/api/statistics/analyze`
2. FastAPI backend receives the request
3. backend calls the LangGraph statistical agent
4. frontend renders the returned answer and structured results inside the existing checker page

### What it does

- Accepts a user question plus a pandas `DataFrame` or file-backed dataset
- Uses LangGraph with these nodes:
  - `parse_request`
  - `inspect_data`
  - `select_method`
  - `run_analysis`
  - `explain_results`
- Uses `ChatOllama(model="gpt-oss:20b", base_url="http://localhost:11434", temperature=0)`
- Uses the LLM only for intent detection and explanation
- Runs deterministic Python code for descriptive statistics, Welch t-tests, and chi-square tests

### Install Python dependencies

```bash
pip install -r requirements-statistical-agent.txt
```

### Run the full app integration

```bash
pip install -r requirements-statistical-agent.txt
npm run backend
npm run dev
```

The frontend runs on `http://localhost:8080` and proxies `/api` requests to the Python backend on `http://localhost:8000`.

### Optional CLI testing

```bash
python -m checkwise_stats.cli --question "Describe the age column" --data path/to/data.csv --show-state
```

The CLI remains available for local testing only. It is no longer the main integration path for the app.

### End-to-end test

1. Start Ollama and confirm `gpt-oss:20b` is available on `http://localhost:11434`
2. Start the backend with `npm run backend`
3. Start the frontend with `npm run dev`
4. Sign in through the existing auth page
5. On the checker page, use the new Statistical Agent panel
6. Try one of the suggested questions such as `Compare score between the control and treatment groups.`
>>>>>>> 7ac1a21 (Adaug agentul statistic)
