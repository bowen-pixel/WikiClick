# Reviews and comparisons

Use this as the reading order instead of opening every historical file.

| Question | Review | Figures |
|---|---|---|
| How do BFS and the original distance model compare? | [Initial results](comparisons/TRAINING_RESULTS.md) | [Raw baseline](../figures/raw/baseline/) |
| What was actually trained at each graph size? | [Training audit](comparisons/TRAINING_AUDIT.md) | [Matched audit charts](../figures/raw/training_audit/) |
| Does next-link ranking help on the small graph? | [1,000-article experiment](comparisons/NEXT_LINK_1000.md) | [Ranking charts](../figures/next_link/ranking_1000/) |
| What happened at million-article scale? | [Completed million-scale results](comparisons/MILLION_RANKING_RESULTS.md) | [Million-scale charts](../figures/next_link/ranking_million/) |
| Did evolutionary search improve our ranker? | [100-journey pilot](comparisons/EVOLUTION_100.md) | [Search dashboard and animation](../figures/evolution/evolution_100/) |

The 1,000-article next-link ranking pipeline improved on the raw-distance baseline in its recorded experiment. Added semantic context had no clear completion gain there, and mistake-focused correction retained the previous checkpoint. At million scale, semantic-guided branching reached more goals under the tested budgets, while BFS retained its shortest-path guarantee; those results are a separate experiment. The evolutionary pilot did not establish a further improvement. Consult each review for its controls and limitations.

## Learning curves and “smoothness”

Compare validation performance, route completion, shortest-route rate, extra clicks, latency, and work. A smoother plotted line is presentation; a lower training loss may still fail to generalize. We have not applied cosmetic smoothing or relabeled it as a measured improvement. `loss_function` charts illustrate the mathematical loss shape; they are not an empirical optimization landscape.

## Other documents

- [Training guide](guides/TRAINING.md)
- [Large distance-model guide](guides/LARGE_TRAINING.md)
- [Fresh graph collection](guides/FRESH_MILLION.md)
- [Million-scale ranking workflow](guides/MILLION_RANKING.md)
- [Project plan](planning/PLAN.md)
- [AI-use record](AI_USAGE.md)
- [Previous project introduction](history/PREVIOUS_README.md)
- [Machine-readable comparisons](results/)
- [File organization and cleanup](organization/README.md)
