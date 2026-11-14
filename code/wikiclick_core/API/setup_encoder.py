"""Explicitly download the pinned encoder; live search itself stays local-only."""
import argparse
from pathlib import Path
from ..paths import ROOT

REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
DEFAULT = ROOT/"data/cache/semantic_encoders/models--google--embeddinggemma-2/snapshots"/REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    path = snapshot_download("google/embeddinggemma-2", revision=REVISION, local_dir=args.output,
                             allow_patterns=["*.json", "*.safetensors", "1_Pooling/*", "2_Normalize/*", "README.md", "*.jinja"])
    print(f"Encoder ready: {path}")


if __name__ == "__main__":
    main()
