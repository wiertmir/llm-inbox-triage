# 📨 LLM Inbox Triage

A small, practical LLM assistant that takes raw text (an email, a support ticket, a message) and returns a **structured analysis**: category, priority, one-line summary, a suggested reply draft, and extracted data (dates, amounts, names, deadlines) as clean JSON.

Built as **Project 1** on my path to becoming an LLM/GenAI Engineer — a focused way to learn LLM APIs, structured output, prompt engineering, and multi-provider code in one useful tool.

```bash
python triage.py sample.txt
python triage.py sample.txt --provider anthropic
```

## ✨ What it does

Given raw text, it returns:

| Field | Description |
|-------|-------------|
| `category` | e.g. urgent / invoice / spam / question / ignore |
| `priority` | 1 (low) – 5 (critical) |
| `summary` | one-sentence summary |
| `suggested_reply` | a draft response |
| `extracted` | structured data: dates, amounts, names, deadlines |

## 🎯 Why this project

It touches everything required at the LLM fundamentals stage:

- **LLM APIs** — OpenAI (Responses API) and Anthropic
- **Structured Outputs** — a Pydantic schema the API *guarantees*, not "please return JSON" + `json.loads` and a prayer
- **Function calling / tools** — optionally act on extracted data (e.g. add a detected meeting to a calendar)
- **Prompt engineering** — a short system prompt describing the *task*, while the schema enforces the *shape*
- **Evals** — a mini eval harness that measures categorization accuracy (the 2026 differentiator)
- **Production hygiene** — error handling, retries, refusal handling, cost awareness
- **Multi-provider** — same logic runs on OpenAI *and* Anthropic

It's also the seed for later projects: add RAG (Project 2) so it knows the context of past mail, then turn it into an agent (Project 3).

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/wiertmir/llm-inbox-triage.git
cd llm-inbox-triage

# 2. Install
pip install -r requirements.txt

# 3. Set your API key(s) — never hardcode them; .env is gitignored
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python triage.py sample.txt
```

## 🛠️ Roadmap

See the [Issues](https://github.com/wiertmir/llm-inbox-triage/issues) tab. MVP first, enhancements later.

**MVP:**
- [ ] Read text from file / stdin
- [ ] **Structured Outputs**: Pydantic `TriageResult` via `responses.parse` (OpenAI) — schema guaranteed, no `json.loads`
- [ ] Works on OpenAI and Anthropic (`--provider` flag)
- [ ] Pretty terminal output + JSON export
- [ ] **Mini-eval**: 10–20 sample messages with expected output, measure categorization accuracy
- [ ] README with example + lessons learned + eval results

**Later:**
- [ ] Function calling: auto-add detected meetings to calendar
- [ ] Batch mode (process a folder of messages)
- [ ] Refusal / safety handling surfaced cleanly

## 📝 License

MIT

---

*Part of my 6-month roadmap to LLM/GenAI Engineer. 🪶*
