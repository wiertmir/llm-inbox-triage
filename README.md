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

- **LLM APIs** — OpenAI and Anthropic
- **Structured output** — reliable JSON, not free text
- **Function calling / tools** — optionally act on extracted data (e.g. add a detected meeting to a calendar)
- **Prompt engineering** — a system prompt that enforces the output contract
- **Production hygiene** — error handling, retries, cost awareness
- **Multi-provider** — same logic runs on OpenAI *and* Anthropic

It's also the seed for later projects: add RAG (Project 2) so it knows the context of past mail, then turn it into an agent (Project 3).

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/wiertmir/llm-inbox-triage.git
cd llm-inbox-triage

# 2. Install
pip install -r requirements.txt

# 3. Set your API key(s)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
python triage.py sample.txt
```

## 🛠️ Roadmap

See the [Issues](https://github.com/wiertmir/llm-inbox-triage/issues) tab. MVP first, enhancements later.

**MVP:**
- [ ] Read text from file / stdin
- [ ] System prompt → structured JSON output
- [ ] Works on OpenAI and Anthropic (`--provider` flag)
- [ ] Pretty terminal output + JSON export
- [ ] README with example + lessons learned

**Later:**
- [ ] Function calling: auto-add detected meetings to calendar
- [ ] Batch mode (process a folder of messages)
- [ ] Simple eval harness (10 sample messages, check categorization)

## 📝 License

MIT

---

*Part of my 6-month roadmap to LLM/GenAI Engineer. 🪶*
