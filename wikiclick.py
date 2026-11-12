"""One entry point for collection, BFS, neural models, experiments, and figures."""
import argparse
import os
import runpy
import sys

from codes.paths import MODULES, ROOT

COMMANDS = {
    "api": "api_endpoint", "collect": "sample_dataset", "validate": "validate_dataset",
    "download": "download_fresh_dump", "import": "import_fresh_dump",
    "bfs": "graph_demo", "compare": "compare_search", "distances": "analyze_all_pairs",
    "benchmark": "benchmark_fresh_scale", "raw-train": "train_model", "raw-scale": "large_training",
    "next-link": "rank_navigate", "next-link-train": "next_link_experiment",
    "training-demo": "demo_next_link", "million-data": "million_semantic_data",
    "million-model": "million_next_link", "evolve": "evolutionary_search",
    "fresh-workflow": "run_fresh_scale", "raw-workflow": "run_large_training",
    "million-workflow": "run_million_ranking", "million-status": "million_ranking_status",
    "plot-raw": "plot_results", "plot-audit": "plot_training_audit",
    "plot-next-link": "plot_next_link", "plot-million": "plot_million_ranking",
    "plot-evolution": "plot_evolutionary_search",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s COMMAND [arguments]",
                                     epilog="Use COMMAND --help for its options. See README.md for examples.")
    parser.add_argument("command", choices=COMMANDS)
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        parser.error(f"Unknown command: {command}")
    module = MODULES[COMMANDS[command]]
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs/.mpl-cache"))
    (ROOT / "logs/.mpl-cache").mkdir(parents=True, exist_ok=True)
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
