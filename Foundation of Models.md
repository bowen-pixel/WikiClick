# Foundation of Models

This document maps mathematical objects and research influences to the public source. Paths below are relative to `code/wikiclick_core/`. A paper-inspired component is not a reproduction of the paper's full model or its reported performance.

## Graph, supervision and exact search

Wikipedia is a directed graph $G=(V,E)$; an article hyperlink has cost 1. The reference distance is $d_G(u,t)$, and the set of optimal actions is

$$A^*(v,t)=\{u:(v,u)\in E,\ d_G(u,t)=d_G(v,t)-1\}.$$

Reverse BFS from $t$ supplies distance labels. Unreachable states retain a distinct missing/sentinel value rather than a zero-distance label. See [BFS](code/wikiclick_core/API/bfs.py) and [training supervision](code/wikiclick_core/model/heuristic/bfs_supervision.py). The guarantee applies to the observed graph. [NetworkX shortest-path documentation](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html) documents the library primitive; [MediaWiki Links](https://www.mediawiki.org/wiki/API:Links) documents the fetched edges.

## SHA256 features, pretrained vectors and regression

The original title vector is a signed hash accumulation over word and character n-grams:

$$x_j=\sum_{q:h(q)=j}\operatorname{sign}(q),\qquad z=x/\max(1,\|x\|_2).$$

This deterministic engineering baseline uses SHA256 for bucket/sign assignment; it does not learn linguistic relationships. See [sha256.py](code/wikiclick_core/model/title_feature/sha256.py). The alternate frozen title vector comes from [EmbeddingGemma-2's model card](https://huggingface.co/google/embeddinggemma-2); see [embeddinggemma.py](code/wikiclick_core/model/title_feature/embeddinggemma.py). Pretraining is upstream work, not WikiClick training.

The distance head receives $[z_u,z_t,|z_u-z_t|,z_u\odot z_t]$, then two ReLU layers and a scalar Softplus output. It minimizes Smooth L1 distance error, for the default unit transition:

$$\ell(e)=\begin{cases}\tfrac12e^2,&|e|<1\\|e|-\tfrac12,&\text{otherwise}.\end{cases}$$

See [network.py](code/wikiclick_core/model/neural/network.py), [pair features](code/wikiclick_core/model/title_feature/pair.py), [trainer](code/wikiclick_core/model/neural/train.py), and the [PyTorch SmoothL1Loss definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html). Figure **LF01** illustrates this function; **LC01/LC02** show measurements, not an optimization landscape.

## Next-link objectives and negative choices

The original ranker uses $s(u,t)=-\hat d(u,t)+r_\theta(u,t)$, where the optional residual uses hashed introductions/category features. For all outgoing candidates $C$ and optimal set $P$, its uniform-positive listwise objective is

$$L_{\mathrm{uniform}}=\log\sum_{u\in C}\exp s_u-\frac1{|P|}\sum_{p\in P}s_p.$$

See [scoring.py](code/wikiclick_core/model/next_link/scoring.py) and [training](code/wikiclick_core/model/next_link/train.py). This explicitly differs from the positive-probability-mass objective used in the later residual model:

$$L_{\mathrm{mass}}=\log\sum_{u\in C}\exp s_u-\log\sum_{p\in P}\exp s_p.$$

See [residual_ranker.py](code/wikiclick_core/model/next_link/residual_ranker.py). Both allow multiple shortest outgoing links; their loss values are not interchangeable.

[Burges et al., *Learning to Rank using Gradient Descent* (2005)](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf) motivates the pairwise logistic preference objective

$$L_{\mathrm{pair}}=\frac1{|P||N|}\sum_{p\in P,n\in N}\log(1+\exp(s_n-s_p)).$$

[negative_sampling.py](code/wikiclick_core/model/next_link/negative_sampling.py) implements this RankNet-style adaptation with optimal/nonoptimal hyperlink labels. Mixed sampling takes up to four high-scoring wrong candidates and four random remaining wrong candidates, without duplicates. [Sun et al., *Ensemble Ranking Model with Multiple Pretraining Strategies for Web Search* (2023)](https://arxiv.org/abs/2302.09340) informed the negative-augmentation experiment; their ensemble, propensity correction and full pretraining system are not implemented here. Pairwise utilities are public; the full historical ablation driver remains local.

## Semantic relevance and context

Live baseline vectors follow

$$z(a)=\operatorname{normalize}(\operatorname{normalize}(E(\mathrm{title}_a))+
\operatorname{normalize}(E(\mathrm{intro}_a))),\qquad s(a,t)=z(a)^\top z(t).$$

Introductions use their first 208 whitespace-separated words; an absent introduction contributes zero. Encoding uses the frozen 128-dimensional SentenceSimilarity representation. See [live_fetch.py](code/wikiclick_core/API/live_fetch.py) and [cosine.py](code/wikiclick_core/model/semantic/cosine.py). Offline title-only regression can use other supported dimensions; it is a separate representation.

[context_ranker.py](code/wikiclick_core/model/semantic/context_ranker.py) retains the later paper-inspired architectures: current/candidate/goal interactions, candidate-set attention, passage similarity pooling, and an auxiliary degree/category residual. Attention uses $\alpha_i=\mathrm{softmax}(q^Tk_i/\sqrt{32})$ and a weighted value summary. The concrete feature tensors are prepared by the local research pipeline; this model is not the default live scorer.

| Source | Specific influence | Scope in WikiClick |
|---|---|---|
| [Zhu et al., Explicit and Implicit Semantic Ranking Framework / sRank](https://arxiv.org/abs/2304.04918) | Semantic query/candidate interaction and context-sensitive ranking | Lightweight context/attention experiments; no sRank Transformer reproduction |
| [Pang et al., DeepRank](https://arxiv.org/abs/1710.05649) | Detect relevant local contexts and aggregate evidence | Local passage-matching residual; no full CNN/2D-GRU architecture |
| [Qi et al., Finding Salient Context](https://arxiv.org/abs/1909.01165) | Select salient text evidence for ranking | Maximum/mean passage similarity features; not their complete sliding-window system |
| [Petrov et al., Robust Fusion of Semantic and Behavioural Signals](https://arxiv.org/abs/2609.25825) | Train with auxiliary features present and removed | `AuxiliaryRanker` supports removal; paired-loss scheduling belongs to the local experiment, not live inference |

## Heuristic correction and progress

The original correction stage weights erroneous **training** decisions by 4 while ordinary decisions have weight 1, then resamples training examples. Validation chooses a checkpoint, including epoch zero; it can reject every attempted update. See [correction.py](code/wikiclick_core/model/heuristic/correction.py) and [next-link trainer](code/wikiclick_core/model/next_link/train.py). This is supervised hard-example mining, not autonomous truth correction, reinforcement learning or test-time learning.

The later residual ranker adds a training-only progress target $q^*=1/(1+d_G)$ for reachable candidates, zero for unreachable ones, a squared-error progress loss and a squared correction penalty. See [progress_head.py](code/wikiclick_core/model/next_link/progress_head.py). BFS labels supervise training/evaluation only; they are not supplied to live inference. Learning from user-supplied optimal live routes is a future extension, not an existing feature.

## Random walks and behavioral cloning

[Zaheer et al., *Learning to Navigate Wikipedia by Taking Random Walks* (2022)](https://arxiv.org/abs/2211.00177) motivates learning goal-conditioned link choices from forward trajectories. WikiClick samples an outgoing edge at each step, selects the eventual endpoint as a goal, terminalizes at its first encounter, and trains action preferences. See [random_walk.py](code/wikiclick_core/model/heuristic/random_walk.py) and [behavioral_cloning.py](code/wikiclick_core/model/semantic/behavioral_cloning.py). The residual policy combines cosine with current/candidate/goal features. WikiClick's graph unit, frozen encoder, sampling controls and search adaptation differ from the paper; its published success rates do not transfer to this project.

The saved corpora also compare shortest-path demonstrations, mixed trajectories and BFS-relabeled walk states. Figure **RW01** is an offline scaling experiment whose selected corpus was shortest-path states, not evidence of a deployed live random-walk policy. Integrating learned walk guidance with live API acquisition remains future work.

## Search and evolutionary selection

[best_first.py](code/wikiclick_core/model/navigator/best_first.py) maintains a frontier ordered by comparable raw scores. Greedy commits locally; beam search retains a bounded set of states. Learned cosine/residual scores are not proven admissible distance lower bounds, so these methods have no A* shortest-path guarantee. Two-link API probes validate actual edges before expensive encoding.

[architecture_search.py](code/wikiclick_core/model/evolutionary/architecture_search.py) uses tournament selection, bounded architecture mutations and retained elites, with an equal-size random-search control and validation rollback. This is the project's engineering search design, not a claimed reproduction of a named evolutionary paper. **EV01/EV02** identify its measured dashboard and architecture schematic. Historical search outcomes did not establish an overall improvement.
