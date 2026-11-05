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
