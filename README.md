# MiniMax-M2.1 Deep Research Agent

A sophisticated research tool powered by **Minimax M2.1** with interleaved thinking, **Exa** neural search, and multi-agent orchestration.

## Features

- **Minimax M2.1 Supervisor**: Uses interleaved thinking to maintain reasoning state across multi-step research
- **Intelligent Planning**: Automatically decomposes research queries into optimized subqueries
- **Neural Web Search**: Leverages Exa API for high-quality, AI-powered web search
- **Comprehensive Reports**: Generates detailed research reports with citations and analysis
- **CLI Interface**: Simple command-line interface with interactive and single-query modes

## Architecture

```
+-----------------------------------------------+
|            Supervisor Agent                   |
|    (Minimax M2.1 + Interleaved Thinking)      |
+-----------------------------------------------+
                      |
       +--------------+--------------+
       |              |              |
       v              v              v
+------------+ +-------------+ +-----------+
|  Planning  | | Web Search  | | Synthesis |
|   Agent    | |  Retriever  | |  (M2.1)   |
|  (Gemini)  | |             | |           |
+------------+ +-------------+ +-----------+
                      |
                      v
               +------------+
               |  Exa API   |
               +------------+
```

### Agent Descriptions

| Agent | Model | Role |
|-------|-------|------|
| **Supervisor** | Minimax M2.1 | Coordinates workflow, synthesizes final report |
| **Planning** | Gemini 2.5 Flash | Generates optimized subqueries |
| **Web Search** | Gemini 2.5 Flash + Exa | Executes searches, organizes findings |

---

## Quick Start

```bash
# Clone and setup
cd deep-research-agent
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Run
uv run python main.py -q "Your research query here"
```

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for:
  - Minimax (M2.1 model)
  - OpenRouter (for Gemini)
  - Exa (web search)

### Setup

1. **Install dependencies**:
```bash
cd deep-research-agent
uv sync
```

2. **Configure environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
MINIMAX_API_KEY=your_minimax_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
EXA_API_KEY=your_exa_api_key_here
```

---

## Usage

### Interactive Mode

```bash
uv run python main.py
```

Then enter your research queries at the prompt.

### Single Query Mode

```bash
uv run python main.py -q "What are the latest developments in quantum computing?"
```

### Save Report to File

```bash
uv run python main.py -q "AI trends in 2025" --save
```

### Verbose Mode

```bash
uv run python main.py -q "Climate change solutions" --verbose
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-q, --query` | Research query (skips interactive mode) |
| `-s, --save` | Save report to `reports/` folder |
| `-v, --verbose` | Show detailed progress and thinking blocks |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/save <query>` | Save the report to a file |
| `/verbose <query>` | Show detailed progress |
| `/help` | Show help message |
| `exit`, `quit`, `q` | Exit the program |

---

## How It Works

### 1. Query Planning

The **Planning Agent** decomposes your query into 3-5 optimized subqueries:

```json
{
  "subqueries": [
    {"query": "quantum computing breakthroughs 2025", "type": "news", "priority": 1},
    {"query": "quantum computing applications cryptography", "type": "auto", "priority": 2}
  ]
}
```

### 2. Web Search

The **Web Search Retriever** executes each subquery using Exa:
- Performs neural search for each subquery
- Finds similar content for high-priority results
- Extracts highlights and key information

### 3. Synthesis

The **Supervisor Agent** (Minimax M2.1):
- Maintains reasoning state via interleaved thinking
- Synthesizes comprehensive report with:
  - Table of contents
  - Key takeaways
  - Executive summary
  - Detailed analysis
  - Cited sources

### 4. Interleaved Thinking

The key innovation: the supervisor preserves ALL content blocks (thinking + text + tool_use) in conversation history. This maintains the reasoning chain across multiple turns for more coherent reports.

---

## Project Structure

```
deep-research-agent/
├── main.py                    # CLI entry point
├── .env.example               # Environment template
├── pyproject.toml             # Dependencies
└── src/
    ├── agents/
    │   ├── supervisor.py           # Minimax M2.1 supervisor
    │   ├── planning_agent.py       # Query planning
    │   └── web_search_retriever.py # Exa search integration
    ├── tools/
    │   └── exa_tool.py             # Exa API wrapper
    └── utils/
        └── config.py               # Configuration
```

---

## API Keys

### Getting API Keys

| Service | URL | Purpose |
|---------|-----|---------|
| Minimax M2.1 | [platform.minimax.io](https://platform.minimax.io) | Supervisor reasoning |
| OpenRouter | [openrouter.ai](https://openrouter.ai) | Planning agent (Gemini) |
| Exa | [exa.ai](https://exa.ai) | Neural web search |

---

## Examples

### Technology Research
```bash
uv run python main.py -q "What are the latest breakthroughs in artificial general intelligence?"
```

### Business Intelligence
```bash
uv run python main.py -q "What are the emerging trends in electric vehicle adoption?" --save
```

### Scientific Research
```bash
uv run python main.py -q "What are the most promising approaches to carbon capture technology?" --verbose
```

---

## Customization

### Adjust Report Style
Edit system prompt in `src/agents/supervisor.py`

### Modify Search Parameters
Edit `src/agents/web_search_retriever.py`:
- `num_results`: Results per query (default: 5-10)
- `time_period`: Date filtering
- `content_type`: Filter by type (news, research, blog)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing API keys | Ensure `.env` exists and has all keys set |
| API errors | Verify keys are valid, check rate limits |
| Import errors | Run `uv sync` and use `uv run python main.py` |

---

## Performance

- Average query time: **30-60 seconds**
- Factors: number of subqueries (3-5), search complexity, LLM response times

---

## License

MIT License

---

## Acknowledgments

Built with:
- [Minimax M2.1](https://www.minimax.io/) - Advanced reasoning model
- [Exa](https://exa.ai/) - Neural web search
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) - API client
- [OpenRouter](https://openrouter.ai/) - LLM routing

---

## Report Grounding Auditor

After the supervisor synthesizes a report, a **grounding auditor** runs an
independent pass that checks the report's inline citations and numeric claims
against the sources actually retrieved by the web search retriever, then
appends a `Source Grounding Audit` section to the report flagging anything it
could not trace back to evidence. This surfaces fabricated or unsupported
citations before the report ships.

The auditor is deterministic and parameter-free (citation-URL matching plus
lexical claim overlap), so it runs on every report with no extra API calls.
Adapted from the Auditor agent in *BrainPilot: Automating Brain Discovery
with Agentic Research* (arXiv:2607.15079) — Mode 2 adapted port, where
BrainPilot's curated knowledge base is replaced by this pipeline's retrieved
Exa sources and its LLM fabrication judge by the parameter-free grounding
proxy. Implementation lives in `src/agents/auditor.py`.

**Scope.** BrainPilot's Auditor audits two dimensions — *evidence backing /
fabrication* and *scientific reliability* (validity defects such as data
leakage or metric misuse in the experiment pipeline). This port implements
only the first dimension; the reliability dimension is deliberately not ported
because it inspects experiment-pipeline artifacts that a general web-research
tool does not produce. The numeric check within Dimension 1 is a lexical proxy
(claim-term overlap with retrieved sources), not the reference's exact-value
grep over workspace artifacts.

---

## Graph of Trace

Alongside the grounding audit, each report carries a **Graph of Trace** — an
auditable record of the workflow that produced it. As the supervisor runs, it
records the research subgoal, every `planning_agent` / `web_search_retriever`
tool call, the evidence each call returned, and the final synthesized report,
then appends a `## Graph of Trace` section rendering those steps as a linked
tree. This lets a reader follow and inspect exactly how a result was reached.

The recorder is deterministic and parameter-free (it captures tool names,
inputs, source counts, and citation counts the supervisor already has), so it
adds no API calls and complements the grounding auditor: the trace shows *how*
the report was built, the auditor checks *whether* its claims are grounded.
Adapted from the Graph of Trace in *BrainPilot: Automating Brain Discovery
with Agentic Research* (arXiv:2607.15079) — Mode 2 adapted port, where
BrainPilot's full graph over PI and specialist agents is replaced by a per-step
trace over this pipeline's own agents. Implementation lives in
`src/agents/research_trace.py`.

---

## Multi-Agent Coordination Analysis

Each report also carries a **coordination analysis** that takes an
information-bottleneck view of the run's multi-agent relays. The supervisor →
planning agent → web search retriever → report chain exchanges information only
through *bounded relay* messages, so the analysis asks the question posed in
*When Do Multi-Agent Systems Help? An Information Bottleneck Perspective*:
did this run's context reduction outweigh its relay information loss? It
reconstructs the relay chain from the Graph of Trace, measures how much
gathered source text the bounded relays kept out of the supervisor's shared
context (a character-based **context-reduction benefit**) against how much
gathered evidence never surfaced in the report (an evidence-based **relay
information loss**), combines them into an effective balance (the paper's
`beta`), and appends a `## Multi-Agent Coordination Analysis` section flagging
the run as favorable, balanced, or lossy.

The analyzer is deterministic and parameter-free (it reads the trace plus the
sources and report the supervisor already has), so it adds no API calls. It
complements the grounding auditor: the auditor checks whether the report's
citations are *grounded* in retrieved sources (precision), while this analysis
measures how much gathered evidence *flowed through* the relays into the report
(recall) — the recall direction is what the paper's relay-sufficiency finding
turns on. Adapted from *When Do Multi-Agent Systems Help? An Information
Bottleneck Perspective* (arXiv:2607.16133) — Mode 2 adapted port, where the
paper's theoretical `beta` and benchmark episode-accounting are replaced by a
parameter-free per-run diagnostic, and its "infinite-bandwidth" single-agent
baseline by an evidence-retention proxy.

**Scope.** The two bottleneck axes are deliberately measured in different
units — context reduction as character compression, information loss as a
fraction of gathered sources dropped — because measuring both off the same
count would collapse them into one number. Both are proxies: the reduction
benefit approximates the transmission cost the bounded relays avoided with raw
source-text size versus report size, and the model-capability term the paper's
`beta` depends on is surfaced only as the supervisor model name, not measured.
Implementation lives in `src/agents/relay_flow.py`.


