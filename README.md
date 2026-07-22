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

## Recursive Deep-and-Wide Search

The web search step is no longer a single wide pass. The **Web Search Retriever**
now drives a recursive delegation tree over Exa: each planned subquery seeds a
search node that couples a local objective with a *search mode* — `atom` (one
precise lookup), `wide` (broad coverage), `deep` (drill into a top hit plus its
neighbors), or `entity_collect` (gather everything about a named entity). A node
runs its mode, then delegates `entity_collect` child nodes for the named-entity
gaps its results surface, up to a recursion depth and a total Exa-call budget.
Evidence aggregates and deduplicates upward, so a single subquery can fan out
into specialized follow-up searches the planner never named — the depth the
single-pass stage was missing.

The tree first probes how task-relevant information is organized on the web (one
wide search that surfaces dominant source domains and named entities) to ground
which entities later nodes collect, and the `retrieve() -> findings` contract is
preserved unchanged: every node returns `search_with_subqueries`-shaped buckets,
so synthesis, the source capture into `last_search_results`, and the grounding
auditor all keep working.

Adapted from *WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide
Web Search* (arXiv:2607.08662) — Mode 2 adapted port, where WebSwarm's LLM-driven
delegation estimator is replaced by a parameter-free entity-gap heuristic (named
entities surfaced in gathered results that no node has covered yet). The
recursive delegation tree, the four search modes, the web-structure probe, and
the upward evidence aggregation are kept at full fidelity. The substitution
mirrors the auditor's parameter-free grounding proxy: deterministic, needs no
extra API keys, and is unit-testable offline. Implementation lives in
`src/agents/recursive_search.py`, wired in through
`src/agents/web_search_retriever.py`.

**Scope.** WebSwarm's "process-level experience reuse across homogeneous sibling
nodes" is deliberately not ported — it is an efficiency optimization, not part of
the core recursive-delegation mechanism. WebSwarm's own benchmark suites
(BrowseComp-Plus, WideSearch, DeepWideSearch, GISA) are evaluation and belong in
a downstream PR; this port ships the search mechanism itself.


