"""Public WikiClick launcher. Run from any working directory."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "live": "API.live_fetch", "bfs": "API.bfs", "collect": "API.training_data",
    "api": "API.wiki_api", "validate": "API.validate_data",
    "prepare-data": "API.data_bundle", "download-encoder": "API.setup_encoder",
    "raw-train": "model.neural.train", "next-link-train": "model.next_link.train",
    "next-link": "model.navigator.saved", "compare": "model.navigator.offline_search",
    "evolve": "model.evolutionary.architecture_search",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, usage="%(prog)s COMMAND [arguments]")
    parser.add_argument("command", choices=COMMANDS)
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        parser.error(f"Unknown command: {command}")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT/"code"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"logs/.mpl-cache"))
    module = "wikiclick_core."+COMMANDS[command]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
