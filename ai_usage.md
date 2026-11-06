# AI usage record

This file records AI assistance used in WikiClick. It distinguishes planning and drafting from implemented and verified functionality. Update it when AI contributes to later code, tests, experiments, or documentation.

## Initial planning and repository documentation

- **Date:** November 4, 2026 (America/Toronto).
- **Tool:** OpenAI Codex, with an earlier planning discussion in the same chat.
- **User request:** Plan a Wikipedia fewest-click project using NetworkX, separate the work into modules and GitHub branches, target December 1, and prepare `README.md`, `Outlin.md`, and `ai_usage.md`.
- **AI contribution:** Explained the directed, unweighted graph model; recommended exact shortest-path search as the baseline; proposed an optional neural-network experiment; drafted the repository overview, module boundaries, milestone schedule, validation goals, and this disclosure. Prepared a Python-oriented `.gitignore` for local/generated files.
- **Sources consulted:** Official NetworkX `shortest_path` and shortest-path algorithm documentation; MediaWiki links, API etiquette, and rate-limit documentation; PyTorch quickstart.
- **Verification at this stage:** Documentation and local-link consistency checks only. No application has been implemented, no Wikipedia graph has been collected, and no model has been trained or evaluated.
- **Human review:** Pending. Repository creation or publication does not imply that the owner has reviewed every technical detail.

The initial illustrative NetworkX snippet is a proposed usage example, not evidence of an implemented application. Performance targets and dataset sizes are planning estimates, not measured results.

## API and shortest-path prototype

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Task:** Use the Wikipedia-API package on a feature branch, extract article links, assign weight 1 to directed edges, and test NetworkX on a small domain.
- **Files:** `api_endpoint.py`, `graph_demo.py`, dependency files, `tests/test_api_and_graph.py`, and documentation updates.
- **Contribution:** Drafted the Python implementation, six offline tests, and setup instructions. Used the package's documented page/link interface; inspected installed package behavior for metadata, pagination, configuration, and exceptions.
- **Validation actually performed:** Installed Wikipedia-API 0.16.0 and NetworkX 3.7 in a local Python 3.13 virtual environment. Six pytest cases passed. Live collection returned five article records; the graph had 13 edges of weight 1. NetworkX found `Graph theory → Shortest path problem → Dijkstra's algorithm` in 2 clicks.
- **Limitations:** Outgoing redirect aliases are matched only when canonical or explicitly requested titles are known. No persistent HTTP cache, full crawler, ML model, or web service was built. Live data is saved locally outside Git. Human review remains pending.

## Ongoing logging template

For each material contribution, add:

```text
Date:
Tool/model (if known):
Task or prompt summary:
Files or pull request affected:
AI contribution:
Human changes and review:
Validation actually performed:
Limitations or unresolved questions:
```

Do not invent model versions, human approvals, test results, or claims that generated work was independently written. If this project is submitted for a course, check the course's actual AI-use rules and adapt this record to its required format.

Do not include passwords, tokens, private chat transcripts, or other sensitive information in this public log. A concise task summary is sufficient unless a specific submission policy requires more.

## Interactive article input

- **Date:** November 5, 2026 (America/Toronto).
- **Tool:** OpenAI Codex.
- **Contribution:** Added source/target prompts to the graph demo and an optional interactive collection mode with intermediate articles. Kept explicit command-line arguments available. Updated both matching local checkouts and documented usage.
- **Validation:** Prompt tests and offline CLI smoke checks; no additional live Wikipedia collection was needed.
- **Limitations:** Interactive input still searches only the saved graph; entering endpoints does not discover routes outside its selected articles. Human review pending.
