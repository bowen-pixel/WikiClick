# Million-article next-link ranking experiment

Completed graph evaluation: 1,001,100 nodes, 14,742,079 directed links.

Same destination-disjoint test decisions and queries for all models. Indexing costs separate; all title/text/category vectors precomputed. Ranking scores guide greedy best-first, not guaranteed optimal. One search prefix supplies budgets 200/2000/10000 with a common 3-second cap. Greedy navigation separately uses 20 clicks. Correction mines only training mistakes; ordinary continuation has equal update count. Identical retained checkpoints reuse measured route outcomes and explicitly marked timings.

## Feature coverage

{
  "graph_id": "sha256:84e3c8034b3b6b797f881f3b04d8b66302c5a72a23b2a40642f63a124399e925",
  "snapshot": "20261101",
  "feature_version": "title-sha256-128+intro-word-crc32-64+explicit-category-pool-64-v1",
  "nodes": 1001100,
  "text_articles_found": 1000970,
  "category_hyperedges": 634833,
  "nonempty_intro_vectors": 999054,
  "articles_in_hyperedges": 928720,
  "scope": "Fixed bounded word-hashed intro features and explicit-category group pooling. Missing text is zero-filled. No pretrained text encoder, learned relation extractor, or learned hypergraph propagation. Categories may include hidden/maintenance categories and omit template-inherited membership. All graph text is visible as transductive unlabeled input."
}

## Matched branching results

Percentages are seed means on one shared query set, not independent repeated test sets. All conditions use the same expansion/time caps. Unsuccessful routes count as non-optimal.

| Model | Budget | Reached | Optimal | Mean expansions | Mean candidates scored | Median ms (seed mean) |
|---|---:|---:|---:|---:|---:|---:|
| Raw distance model | 200 | 13.54% | 0.69% | 179.1 | 4812.8 | 20.44 |
| Raw distance model | 2,000 | 25.00% | 0.69% | 1620.3 | 25272.5 | 156.90 |
| Raw distance model | 10,000 | 31.25% | 0.69% | 7210.3 | 72393.3 | 616.31 |
| Title next-link ranker | 200 | 9.72% | 1.04% | 191.0 | 8603.1 | 30.26 |
| Title next-link ranker | 2,000 | 27.08% | 1.74% | 1661.7 | 33835.0 | 185.49 |
| Title next-link ranker | 10,000 | 31.25% | 1.74% | 7265.9 | 83115.7 | 667.44 |
| Text/category ranker | 200 | 14.58% | 0.00% | 190.0 | 11818.0 | 61.45 |
| Text/category ranker | 2,000 | 41.67% | 3.47% | 1386.9 | 37874.0 | 340.41 |
| Text/category ranker | 10,000 | 56.25% | 3.47% | 5220.8 | 72361.4 | 646.08 |
| Ordinary continuation | 200 | 16.67% | 0.35% | 190.8 | 11816.0 | 61.71 |
| Ordinary continuation | 2,000 | 41.67% | 4.86% | 1370.3 | 37509.3 | 343.38 |
| Ordinary continuation | 10,000 | 56.25% | 4.86% | 5244.8 | 72397.0 | 699.91 |
| Correction with rollback | 200 | 14.58% | 0.00% | 190.0 | 11818.0 | 61.45 |
| Correction with rollback | 2,000 | 41.67% | 3.47% | 1386.9 | 37874.0 | 340.41 |
| Correction with rollback | 10,000 | 56.25% | 3.47% | 5220.8 | 72361.4 | 646.08 |
| BFS | 200 | 2.08% | 2.08% | 198.7 | 0 (no neural scoring) | 1.52 |
| BFS | 2,000 | 9.38% | 9.38% | 1900.3 | 0 (no neural scoring) | 11.98 |
| BFS | 10,000 | 22.92% | 22.92% | 8534.6 | 0 (no neural scoring) | 47.10 |

## Correction checkpoint selection

Epoch 0 means attempted updates were rejected and the previous semantic checkpoint was retained.

| Seed | Stage | Selected epoch | Selected parameter change (L2) |
|---:|---|---:|---:|
| 207 | continued | 1 | 1.217344 |
| 207 | corrected | 0 | 0.000000 |
| 208 | continued | 1 | 1.231797 |
| 208 | corrected | 0 | 0.000000 |
| 209 | continued | 0 | 0.000000 |
| 209 | corrected | 0 | 0.000000 |

## Limits

This is sampled supervision on a million-node graph, not supervision of every article or every pair. The semantic representation uses fixed word-hashed lead text and explicit-category pooling, not a pretrained language model or learned hypergraph propagation. Category metadata differs from the small pilot: wikitext-only categories omit inherited memberships and can include hidden/maintenance categories. A graph snapshot is a collection of revisions, not an atomic instant.

The raw model receives up to 60,000 distance pairs. Rankers have additional training; semantic rankers have an extra residual head. These differences prevent attributing the raw-versus-refined comparison to the loss alone. Ordinary continuation is the equal-update control for mistake resampling. No test labels, test failures or exact distances are input to the search policy. Models are selected by validation loss before test evaluation. No results are extrapolated from the 1,000-article run.

Branching here means prioritizing the frontier: valid outgoing links and graph edge costs are unchanged. Fewer expansions do not imply better search if the method reaches fewer goals. Only BFS has the shortest-path guarantee. Deadline enforcement is checked between bounded batches; a running tensor operation can slightly overshoot. Identical selected checkpoints reuse clearly marked evaluation records rather than fabricate independent timing runs.
