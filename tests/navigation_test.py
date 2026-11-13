"""WikiClick final held-out navigation evaluation; no training or model selection.

Place at codes/models/raw/navigation_test.py. Requires saved outputs from
layer_dropout_experiment.py, the local ranking_1000 graph and frozen encoder.

Primary (preselected) candidate: R7_equal_020. R0 and R1 are references only.
Do not use the test results to retune dropout or choose another winner.
"""
import argparse
import json
import statistics
from pathlib import Path

import networkx as nx
import torch

from codes.models.raw.ml_model import (
    ENCODER_NAME, SEMANTIC_FEATURE_VERSION,
    build_title_vectors, create_encoder, pair_features,
)
from codes.models.raw.layer_dropout_experiment import RegularizedDistanceModel
from codes.models.raw.train_model import read_graph, split_pairs, make_tensors, mean_absolute_error
from codes.search.compare_search import choose_queries, search

DEFAULT_SETTINGS = ('R0_baseline', 'R1_equal_010', 'R7_equal_020')


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_saved_model(path, graph_id, dimensions, hidden, seed, method):
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    # Checkpoints are locally trained, tensor-only files. Never load untrusted weights.
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'R0_baseline': (0.0, 0.0), 'R1_equal_010': (0.1, 0.1),
                'R7_equal_020': (0.2, 0.2)}
    p1, p2 = expected[method]
    for field, target in (('graph_id', graph_id), ('dimensions', dimensions),
                          ('hidden', hidden), ('model_seed', seed),
                          ('feature_method', 'semantic'),
                          ('feature_version', SEMANTIC_FEATURE_VERSION),
                          ('dropout1', p1), ('dropout2', p2),
                          ('data_seed', 611)):
        if ckpt.get(field) != target:
            raise ValueError(f'{path}: {field}={ckpt.get(field)!r}, expected {target!r}')
    model = RegularizedDistanceModel(dimensions, hidden, p1, p2)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
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
    regret_total = 0.0
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
                regret_total += true_selected - best_true
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

    route_by_budget = {}
    for budget in budgets:
        rows = []
        for q in queries:
            outcome = search(graph, q['source'], q['target'], budget, scorer)
            success = outcome['path'] is not None
            if success and not all(graph.has_edge(a, b) for a, b in zip(outcome['path'], outcome['path'][1:])):
                raise AssertionError('Invalid path returned by search')
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
            'one_step_rows': one_step_rows,
            'by_budget': route_by_budget}


def evaluate_bfs(graph, queries, budgets):
    report = {}
    for budget in budgets:
        rows = []
        for q in queries:
            result = search(graph, q['source'], q['target'], budget)
            success = result['path'] is not None
            rows.append({'found': success, 'clicks': result['clicks'], 'expanded': result['expanded'],
                         'seconds': result['seconds'], 'optimal_clicks': q['optimal_clicks']})
        successes = [r for r in rows if r['found']]
        report[str(budget)] = {
            'success_rate': len(successes) / len(rows),
            'optimal_route_rate_all_queries': sum(r['found'] and r['clicks'] == r['optimal_clicks'] for r in rows) / len(rows),
            'mean_expanded_all_queries': statistics.mean(r['expanded'] for r in rows),
            'median_ms': 1000 * statistics.median(r['seconds'] for r in rows),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/graphs/ranking_1000'))
    parser.add_argument('--checkpoints', type=Path, default=Path('models/experiments/layer_dropout_1000'))
    parser.add_argument('--output', type=Path, default=Path('models/experiments/navigation_test_1000/report.json'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[207, 208, 209])
    parser.add_argument('--models', nargs='+', choices=DEFAULT_SETTINGS, default=list(DEFAULT_SETTINGS))
    parser.add_argument('--queries', type=int, default=100)
    parser.add_argument('--budgets', type=int, nargs='+', default=[25, 50, 100, 200])
    parser.add_argument('--encoder-revision')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; select a new --output to prevent accidental overwrite')
    if args.queries < 1 or any(b < 1 for b in args.budgets) or not args.seeds:
        parser.error('Positive queries/budgets and at least one seed required')
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
    print('Preparing held-out test destination pairs...', flush=True)
    test_pairs, test_counts = split_pairs(args.data, 'test', graph_info, goals, graph,
                                         config['max_pairs_per_split'], config['data_seed'])
    print('Encoding semantic article titles once...', flush=True)
    encoder = create_encoder(revision=args.encoder_revision,
                             cache_folder='data/cache/semantic_encoders')
    vectors = build_title_vectors(graph_info, encoder, dimensions, 'semantic')
    del encoder
    test_xy = make_tensors(test_pairs, vectors)
    # Fixed query seed 902 independent from model seeds, same 100 queries for all.
    queries = choose_queries(graph, goals['test'], args.queries, 902)
    bfs = evaluate_bfs(graph, queries, args.budgets)
    results = {}
    for seed in args.seeds:
        results[str(seed)] = {}
        for method in args.models:
            ckpt_path = args.checkpoints / f'seed_{seed}' / method / 'best.pt'
            print(f'Evaluating (no training): seed={seed}, model={method}', flush=True)
            model = load_saved_model(ckpt_path, graph_info['graph_id'], dimensions, hidden,
                                     seed, method)
            result = evaluate_model(model, queries, args.budgets, graph, vectors, test_xy)
            results[str(seed)][method] = result
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
                } for b in args.budgets
            }
        }
    report = {
        'experiment': 'Final held-out destination navigation evaluation',
        'locked_primary_candidate': 'R7_equal_020',
        'note': 'No training. Test results are descriptive, not for retuning hyperparameters.',
        'graph_id': graph_info['graph_id'], 'test_pair_counts': test_counts,
        'queries': queries, 'budgets': args.budgets, 'seeds': args.seeds,
        'bfs': bfs, 'summary': summary, 'results': results,
    }
    save_json(args.output, report)
    print('\nSummary across seeds:')
    print(json.dumps({'bfs': bfs, 'models': summary}, indent=2))
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
