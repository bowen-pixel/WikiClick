"""WikiClick final held-out navigation evaluation; no training or model selection.

Requires completed layer_dropout_1000_v3 checkpoints and a local frozen encoder.

Primary (preselected) candidate: R7_strong_dropout. R0 and R1 are references only.
Do not use the test results to retune dropout or choose another winner.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from codes.models.raw.ml_model import (
    ENCODER_NAME, SEMANTIC_FEATURE_VERSION,
    build_title_vectors, pair_features,
)
from codes.models.raw.regularization_experiment import RegularizedDistanceModel
from codes.models.raw.train_model import read_graph, split_pairs, make_tensors, mean_absolute_error
from codes.search.compare_search import choose_queries, search

EXPECTED = {'R0_baseline': (0.0, 0.0), 'R1_dropout': (0.1, 0.0),
            'R7_strong_dropout': (0.2, 0.2)}
DEFAULT_SETTINGS = tuple(EXPECTED)


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_saved_model(path, graph_id, dimensions, hidden, seed, method):
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    # Checkpoints are locally trained, tensor-only files. Never load untrusted weights.
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    p1, p2 = EXPECTED[method]
    for field, target in (('graph_id', graph_id), ('dimensions', dimensions),
                          ('hidden', hidden), ('model_seed', seed),
                          ('feature_method', 'semantic'),
                          ('feature_version', SEMANTIC_FEATURE_VERSION),
                          ('encoder_name', ENCODER_NAME), ('weight_decay', 0.0),
                          ('dropout1', p1), ('dropout2', p2),
                          ('data_seed', 611)):
        if ckpt.get(field) != target:
            raise ValueError(f'{path}: {field}={ckpt.get(field)!r}, expected {target!r}')
    model = RegularizedDistanceModel(dimensions, hidden, p1, p2)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model


def evaluate_model(model, queries, budgets, graph, vectors, test_xy):
    model.eval()
    x_test, y_test = test_xy
    test_mae = mean_absolute_error(model, x_test, y_test)
    by_target = {}
    for q in queries:
        by_target.setdefault(q['target'], []).append(q)

    # Exact distances are evaluation labels ONLY. Never passed into the scorer.
    optimal_next = correct_next = 0
    regrets, unreachable_selections = [], 0
    one_step_rows = []
    for target, qs in by_target.items():
        distances = nx.single_source_shortest_path_length(graph.reverse(copy=False), target)
        for q in qs:
            source = q['source']
            neighbors = sorted(graph.successors(source))
            if not neighbors:
                continue
            features = torch.stack([pair_features(vectors[n], vectors[target]) for n in neighbors])
            with torch.inference_mode():
                scores = model(features).tolist()
            selected_index = min(range(len(neighbors)), key=lambda idx: (scores[idx], neighbors[idx]))
            selected = neighbors[selected_index]
            # A source may link to unreachable nodes; count these as non-optimal.
            best_true = min((distances.get(n, float('inf')) for n in neighbors), default=float('inf'))
            true_selected = distances.get(selected, float('inf'))
            if best_true == float('inf'):
                continue
            optimal_next += 1
            is_optimal = true_selected == best_true
            correct_next += int(is_optimal)
            if true_selected != float('inf'):
                regrets.append(true_selected - best_true)
            else:
                unreachable_selections += 1
            one_step_rows.append({'source': source, 'target': target,
                                  'selected': selected, 'optimal': is_optimal,
                                  'selected_remaining_clicks': None if true_selected == float('inf') else true_selected,
                                  'optimal_remaining_clicks': best_true})

    @torch.inference_mode()
    def scorer(nodes, target):
        if not nodes:
            return []
        batch = torch.stack([pair_features(vectors[n], vectors[target]) for n in nodes])
        return model(batch).tolist()

    if queries:
        scorer([queries[0]['source']], queries[0]['target'])
    route_by_budget = {}
    for budget in budgets:
        rows = []
        for q in queries:
            outcome = search(graph, q['source'], q['target'], budget, scorer)
            success = outcome['path'] is not None
            if success and not all(graph.has_edge(a, b) for a, b in zip(outcome['path'], outcome['path'][1:])):
                raise AssertionError('Invalid path returned by search')
            if success and (outcome['path'][0] != q['source'] or outcome['path'][-1] != q['target']
                            or outcome['clicks'] < q['optimal_clicks']):
                raise AssertionError('Path endpoints or reference distance differ')
            rows.append({'source': q['source'], 'target': q['target'],
                         'optimal_clicks': q['optimal_clicks'],
                         'found': success, 'clicks': outcome['clicks'],
                         'expanded': outcome['expanded'], 'seconds': outcome['seconds'],
                         'path': outcome['path']})
        successes = [r for r in rows if r['found']]
        route_by_budget[str(budget)] = {
            'success_rate': len(successes) / len(rows),
            'optimal_route_rate_all_queries': sum(r['found'] and r['clicks'] == r['optimal_clicks'] for r in rows) / len(rows),
            'mean_extra_clicks_successes_only': statistics.mean(r['clicks'] - r['optimal_clicks'] for r in successes) if successes else None,
            'mean_expanded_all_queries': statistics.mean(r['expanded'] for r in rows),
            'median_ms': 1000 * statistics.median(r['seconds'] for r in rows),
            'routes': rows,
        }
    return {'test_mae': test_mae,
            'optimal_next_link_accuracy': correct_next / optimal_next if optimal_next else None,
            'evaluated_next_link_cases': optimal_next,
            'unreachable_next_link_selections': unreachable_selections,
            'mean_next_link_regret_reachable_selections_only': statistics.mean(regrets) if regrets else None,
            'one_step_rows': one_step_rows,
            'by_budget': route_by_budget}


def evaluate_bfs(graph, queries, budgets):
    report = {}
    for budget in budgets:
        rows = []
        for q in queries:
            result = search(graph, q['source'], q['target'], budget)
            success = result['path'] is not None
            if success and result['clicks'] != q['optimal_clicks']:
                raise AssertionError('BFS disagrees with exact reference distance')
            rows.append({**q, 'found': success, 'path': result['path'], 'clicks': result['clicks'], 'expanded': result['expanded'],
                         'seconds': result['seconds'], 'optimal_clicks': q['optimal_clicks']})
        successes = [r for r in rows if r['found']]
        report[str(budget)] = {
            'success_rate': len(successes) / len(rows),
            'optimal_route_rate_all_queries': sum(r['found'] and r['clicks'] == r['optimal_clicks'] for r in rows) / len(rows),
            'mean_expanded_all_queries': statistics.mean(r['expanded'] for r in rows),
            'median_ms': 1000 * statistics.median(r['seconds'] for r in rows),
            'routes': rows,
        }
    return report


def load_local_vectors(graph, dimensions, snapshot, runtime_cache):
    """Read an existing encoder snapshot; no downloads or non-raw cache writes."""
    if not (snapshot / 'modules.json').is_file():
        raise FileNotFoundError(f'No complete local sentence encoder at {snapshot}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(runtime_cache)
    os.environ['HF_MODULES_CACHE'] = str(runtime_cache / 'modules')
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(str(snapshot), device='cpu', local_files_only=True,
                                  cache_folder=str(runtime_cache),
                                  model_kwargs={'torch_dtype': torch.float32},
                                  config_kwargs={'vision_config': None, 'audio_config': None})
    encoder.eval().requires_grad_(False)
    vectors = build_title_vectors(graph, encoder, dimensions, 'semantic')
    del encoder
    return vectors


def paired_route_gain(results, reference, candidate, budget, seeds):
    """Paired destination bootstrap, averaging model seeds within each query."""
    grouped = {}
    for seed in seeds:
        a = results[str(seed)][reference]['by_budget'][str(budget)]['routes']
        b = results[str(seed)][candidate]['by_budget'][str(budget)]['routes']
        if [(r['source'], r['target']) for r in a] != [(r['source'], r['target']) for r in b]:
            raise ValueError('Unmatched query identities')
        for x, y in zip(a, b):
            grouped.setdefault(x['target'], []).append([
                int(y['found']) - int(x['found']),
                int(y['found'] and y['clicks'] == y['optimal_clicks']) -
                int(x['found'] and x['clicks'] == x['optimal_clicks'])])
    cluster_means = np.array([np.mean(v, axis=0) for v in grouped.values()])
    indices = np.random.default_rng(1204).integers(0, len(cluster_means), (5000, len(cluster_means)))
    bootstrap = cluster_means[indices].mean(axis=1) * 100
    return {name: {'gain_percentage_points': float(cluster_means[:, i].mean() * 100),
                   'conditional_95_percent_interval': np.quantile(bootstrap[:, i], [.025, .975]).tolist(),
                   'destination_clusters': len(grouped)}
            for i, name in enumerate(['success', 'optimal_route'])}


def write_artifacts(report, output, figures):
    """Only raw-model evaluation outputs; no changes to shared plotters/reviews."""
    os.environ.setdefault('MPLCONFIGDIR', str(figures / '.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True, exist_ok=True)
    methods, budgets = list(report['summary']), report['budgets']
    labels = {name: name.split('_')[0] for name in methods}
    colors = dict(zip(methods, ['#64748b', '#2563eb', '#0d9488']))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    x = np.arange(len(methods))
    for ax, field, title, scale in [
        (axes[0, 0], 'mean_test_mae', 'Held-out distance MAE (lower is better)', 1),
        (axes[0, 1], 'mean_optimal_next_link_accuracy', 'Optimal next-link accuracy (%)', 100)]:
        bars = ax.bar(x, [scale * report['summary'][m][field] for m in methods], color=list(colors.values()))
        ax.bar_label(bars, fmt='%.3f' if scale == 1 else '%.1f', padding=4)
        ax.set(xticks=x, xticklabels=list(labels.values()), title=title)
        ax.margins(y=.2)
    for ax, field, title in [(axes[1, 0], 'mean_success_rate', 'Goal reached (%)'),
                              (axes[1, 1], 'mean_optimal_route_rate_all_queries', 'Minimum-click route (%)')]:
        for method in methods:
            ax.plot(budgets, [100 * report['summary'][method]['per_budget'][str(b)][field] for b in budgets],
                    'o-', label=labels[method], color=colors[method])
        bfs_field = 'success_rate' if field == 'mean_success_rate' else 'optimal_route_rate_all_queries'
        ax.plot(budgets, [100 * report['bfs'][str(b)][bfs_field] for b in budgets],
                's--', label='BFS', color='#b45309')
        ax.set(title=title, xlabel='Expansion budget', xticks=budgets, ylim=(0, 105))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle('Experiment 4 | saved distance models as navigation heuristics', fontsize=16)
    fig.supxlabel('R0: dropout (0,0) · R1: (0.1,0) · R7: (0.2,0.2)\n'
                  '100 identical queries · three model seeds · no training or test-based selection', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(figures / f'navigation_comparison.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Experiment 4: navigation evaluation', '', report['note'], '',
             'R7 is the preselected primary candidate. R0 and R1 are references.',
             'R1 is the saved latest-run **(0.1,0)** model, as explicitly requested after inspecting the artifacts; '
             'it is not the unavailable (0.1,0.1) model described in the initial pasted plan.', '',
             '| Model | Test MAE | Optimal next link |', '|---|---:|---:|']
    for name, row in report['summary'].items():
        lines.append(f"| {name} | {row['mean_test_mae']:.4f} | {100*row['mean_optimal_next_link_accuracy']:.2f}% |")
    lines += ['', '| Budget | Method | Reached | Shortest route | Mean expanded | Extra clicks, successes only | Median ms |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for budget in budgets:
        for name in methods:
            row = report['summary'][name]['per_budget'][str(budget)]
            lines.append(f"| {budget} | {labels[name]} | {100*row['mean_success_rate']:.2f}% | "
                         f"{100*row['mean_optimal_route_rate_all_queries']:.2f}% | {row['mean_expanded_all_queries']:.2f} | "
                         f"{str(round(row['mean_extra_clicks_successes_only'], 3)) if row['mean_extra_clicks_successes_only'] is not None else 'n/a'} | {row['mean_median_ms']:.3f} |")
        row = report['bfs'][str(budget)]
        lines.append(f"| {budget} | BFS | {100*row['success_rate']:.2f}% | "
                     f"{100*row['optimal_route_rate_all_queries']:.2f}% | {row['mean_expanded_all_queries']:.2f} | 0 | {row['median_ms']:.3f} |")
    lines += ['', '## Paired R7 comparisons', '',
              'Intervals are paired bootstrap intervals by destination, conditional on this graph and these trained seeds. '
              'They are exploratory per-budget intervals, without correction for testing multiple budgets.']
    for reference, by_budget in report['paired_R7_vs_references'].items():
        for budget, metrics in by_budget.items():
            s, o = metrics['success'], metrics['optimal_route']
            si = '[' + ', '.join(f'{v:+.2f}' for v in s['conditional_95_percent_interval']) + ']'
            oi = '[' + ', '.join(f'{v:+.2f}' for v in o['conditional_95_percent_interval']) + ']'
            lines.append(f"- R7 − {reference.split('_')[0]}, budget {budget}: completion {s['gain_percentage_points']:+.2f} pp "
                         f"{si}; shortest routes {o['gain_percentage_points']:+.2f} pp {oi}.")
    lines += ['', '## Protocol and limits', '',
              '- Checkpoint metadata, graph, splits, data seed, pair hashes, and validation MAE were checked before test evaluation.',
              '- Features use the existing local encoder snapshot; no downloads or source-cache writes. The original training metadata '
              'did not pin its encoder revision. Matching all saved validation errors is a compatibility check, not proof of original revision identity.',
              '- All methods use the same queries, neighbors, discovery stopping rule, and expansion limits. Dropout is disabled during evaluation.',
              '- BFS distances are evaluation labels, never neural scorer inputs. Budget-limited BFS may fail to reach a known reachable goal.',
              '- Shared-graph destination-held-out evaluation; these test destinations and query seed were used in prior studies. This is not a pristine new benchmark.',
              '- Lower MAE or fewer expansions alone does not establish better navigation. No dropout or checkpoint was reselected using these outcomes.',
              '- Seed means share one query set; they are not independent repetitions of the test set. Timings include inference and exclude encoding.',
              '- Checkpoint files and in-memory weights were verified unchanged after evaluation.', '',
              '## Verification', '',
              'Four focused regression tests live beside this evaluator in `codes/models/raw/test_navigation_evaluation.py`. '
              'They cover nonoptimal routes, unreachable next-link choices, equal optimal choices, dropout metadata/evaluation mode, '
              'and paired query matching. All four passed. The 55 existing regression tests also passed with the standalone '
              '`tests/navigation_test.py` draft excluded: that pre-existing duplicate imports the absent '
              '`codes.models.raw.layer_dropout_experiment` module. It was left untouched to honor the raw-only source scope.', '',
              '## Reproduce', '', '```bash',
              '.venv/bin/python -m codes.models.raw.navigation_test --output models/raw/navigation_test_1000_repeat/report.json '
              '--figures figures/raw/navigation_test_1000_repeat', '```', '',
              'The full report retains exact queries, per-query paths, per-seed outcomes, confidence intervals, and file hashes.']
    output.with_name('README.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/graphs/ranking_1000'))
    parser.add_argument('--checkpoints', type=Path, default=Path('models/experiments/layer_dropout_1000_v3'))
    parser.add_argument('--output', type=Path, default=Path('models/raw/navigation_test_1000/report.json'))
    parser.add_argument('--figures', type=Path, default=Path('figures/raw/navigation_test_1000'))
    parser.add_argument('--encoder-cache', type=Path, default=Path('data/cache/semantic_encoders/models--google--embeddinggemma-2'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[207, 208, 209])
    parser.add_argument('--models', nargs='+', choices=DEFAULT_SETTINGS, default=list(DEFAULT_SETTINGS))
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--budgets', type=int, nargs='+', default=[25, 50, 100, 200])
    parser.add_argument('--encoder-revision')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; select a new --output to prevent accidental overwrite')
    if args.queries < 1 or not args.budgets or any(b < 1 for b in args.budgets) or not args.seeds:
        parser.error('Positive queries/budgets and at least one seed required')
    if set(args.models) != set(DEFAULT_SETTINGS) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Evaluate exactly R0/R1/R7 with distinct seeds; no test-driven candidate selection')
    args.budgets = sorted(set(args.budgets))
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

    graph_info, goals, graph = read_graph(args.data)
    config_path = args.checkpoints / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['graph_id'] != graph_info['graph_id'] or config.get('data_seed') != 611:
        raise ValueError('Checkpoint experiment graph or data seed differs')
    dimensions, hidden = config['dimensions'], config['hidden']
    # Require an existing completed validation study; do not select winners here.
    comparison_path = args.checkpoints / 'comparison.json'
    if not comparison_path.is_file():
        raise FileNotFoundError('Missing comparison.json; finish validation experiment before final test')
    comparison = json.loads(comparison_path.read_text())
    if comparison.get('provisional_validation_winner') != 'R7_strong_dropout':
        raise ValueError('Expected completed validation study with preselected R7')
    loaded, provenance = {}, {}
    for seed in args.seeds:
        loaded[str(seed)], provenance[str(seed)] = {}, {}
        for method in args.models:
            if config['settings'][method] != dict(dropout1=EXPECTED[method][0], dropout2=EXPECTED[method][1], weight_decay=0.0):
                raise ValueError(f'Experiment configuration mismatch for {method}')
            path = args.checkpoints / f'seed_{seed}' / method / 'best.pt'
            loaded[str(seed)][method] = load_saved_model(path, graph_info['graph_id'], dimensions, hidden, seed, method)
            provenance[str(seed)][method] = {'path': str(path), 'sha256': file_hash(path),
                                             'state_sha256': state_hash(loaded[str(seed)][method])}
    revision = args.encoder_revision or (args.encoder_cache / 'refs/main').read_text().strip()
    snapshot = args.encoder_cache / 'snapshots' / revision
    print(f'Encoding titles once, entirely offline: {revision}', flush=True)
    feature_started = time.perf_counter()
    vectors = load_local_vectors(graph_info, dimensions, snapshot, args.output.parent / '.encoder_cache')
    feature_seconds = time.perf_counter() - feature_started
    feature_hash = hashlib.sha256(torch.stack([vectors[n['pageid']] for n in graph_info['nodes']]).numpy().tobytes()).hexdigest()
    validation_pairs, validation_counts = split_pairs(args.data, 'validation', graph_info, goals, graph,
                                                       config['max_pairs_per_split'], config['data_seed'])
    if validation_counts != config['counts']['validation']:
        raise ValueError('Validation pairs differ from the training experiment')
    validation_xy = make_tensors(validation_pairs, vectors)
    for seed in args.seeds:
        for method in args.models:
            actual = mean_absolute_error(loaded[str(seed)][method], *validation_xy)
            expected = comparison['results'][str(seed)][method]['validation_mae']
            if abs(actual - expected) > 1e-5:
                raise ValueError(f'Encoder/checkpoint compatibility failed: {seed}/{method}: {actual} != {expected}')
            provenance[str(seed)][method].update(saved_validation_mae=expected, reproduced_validation_mae=actual)
    del validation_xy
    print('All saved validation MAEs reproduced. Opening test evaluation...', flush=True)
    test_pairs, test_counts = split_pairs(args.data, 'test', graph_info, goals, graph,
                                         config['max_pairs_per_split'], config['data_seed'])
    test_xy = make_tensors(test_pairs, vectors)
    # Fixed query seed 902 independent from model seeds, same 100 queries for all.
    queries = choose_queries(graph, goals['test'], args.queries, 902)
    if len(queries) != args.queries or len({q['target'] for q in queries}) != args.queries:
        raise ValueError('Expected exactly one reachable query per requested held-out destination')
    bfs = evaluate_bfs(graph, queries, args.budgets)
    results = {}
    for seed in args.seeds:
        results[str(seed)] = {}
        for method in args.models:
            print(f'Evaluating (no training): seed={seed}, model={method}', flush=True)
            model = loaded[str(seed)][method]
            result = evaluate_model(model, queries, args.budgets, graph, vectors, test_xy)
            results[str(seed)][method] = result
            evidence = provenance[str(seed)][method]
            if state_hash(model) != evidence['state_sha256'] or file_hash(Path(evidence['path'])) != evidence['sha256']:
                raise AssertionError('Evaluation modified checkpoint weights')
            evidence['unchanged_after_evaluation'] = True
            print(f"  test MAE={result['test_mae']:.4f} | next-link top1={result['optimal_next_link_accuracy']:.1%}")

    summary = {}
    for method in args.models:
        items = [results[str(seed)][method] for seed in args.seeds]
        summary[method] = {
            'mean_test_mae': statistics.mean(item['test_mae'] for item in items),
            'mean_optimal_next_link_accuracy': statistics.mean(item['optimal_next_link_accuracy'] for item in items),
            'per_budget': {
                str(b): {
                    'mean_success_rate': statistics.mean(item['by_budget'][str(b)]['success_rate'] for item in items),
                    'mean_optimal_route_rate_all_queries': statistics.mean(item['by_budget'][str(b)]['optimal_route_rate_all_queries'] for item in items),
                    'mean_expanded_all_queries': statistics.mean(item['by_budget'][str(b)]['mean_expanded_all_queries'] for item in items),
                    'mean_extra_clicks_successes_only': (statistics.mean(values) if (values := [item['by_budget'][str(b)]['mean_extra_clicks_successes_only'] for item in items if item['by_budget'][str(b)]['mean_extra_clicks_successes_only'] is not None]) else None),
                    'mean_median_ms': statistics.mean(item['by_budget'][str(b)]['median_ms'] for item in items),
                } for b in args.budgets
            }
        }
    report = {
        'experiment': 'Final held-out destination navigation evaluation',
        'locked_primary_candidate': 'R7_strong_dropout',
        'note': 'No training or checkpoint selection. Previously evaluated test destinations/query seed; '
                'results are descriptive, not a pristine new holdout or a basis to retune dropout.',
        'graph_id': graph_info['graph_id'], 'test_pair_counts': test_counts,
        'queries': queries, 'budgets': args.budgets, 'seeds': args.seeds,
        'bfs': bfs, 'summary': summary, 'results': results,
        'dropout_probabilities': {k: list(v) for k, v in EXPECTED.items()},
        'checkpoint_provenance': provenance,
        'encoder': {'name': ENCODER_NAME, 'local_snapshot': str(snapshot), 'revision_used': revision,
                    'training_revision_unrecorded': config.get('encoder_revision_requested') is None,
                    'feature_sha256': feature_hash, 'encoding_seconds': feature_seconds},
        'paired_R7_vs_references': {reference: {str(b): paired_route_gain(results, reference, 'R7_strong_dropout', b, args.seeds)
                                               for b in args.budgets} for reference in ['R0_baseline', 'R1_dropout']},
        'elapsed_seconds_excluding_plots': time.perf_counter() - started,
    }
    save_json(args.output, report)
    write_artifacts(report, args.output, args.figures)
    print('\nSummary across seeds:')
    print(json.dumps({'bfs': {b: {k: v for k, v in row.items() if k != 'routes'} for b, row in bfs.items()}, 'models': summary}, indent=2))
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
