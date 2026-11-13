"""Title features and the shared distance head used by both comparison arms."""
import hashlib
import re

import torch
from torch import nn

ENCODER_NAME = "google/embeddinggemma-2"
# Retain the original identifier/defaults for existing checkpoints and rankers.
FEATURE_VERSION = "title-char-3-5-and-words-sha256-v1"
SEMANTIC_FEATURE_VERSION = "embeddinggemma-2-title-v1"


def title_features(title, dimensions=128):
    """Original, deterministic signed SHA256 word/character hashing."""
    if dimensions < 1:
        raise ValueError("Feature dimension must be positive")
    text = " " + " ".join(title.lower().split()) + " "
    tokens = re.findall(r"\w+", text)
    tokens += [text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1))]
    result = torch.zeros(dimensions)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "little") % dimensions] += 1 if digest[4] % 2 else -1
    return result / result.norm().clamp_min(1)


def create_encoder(model_name=ENCODER_NAME, revision=None, cache_folder=None):
    """Load the optional, frozen semantic encoder only when requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Semantic features require: .venv/bin/python -m pip install sentence-transformers") from error
    kwargs = {"device": "cpu", "revision": revision, "cache_folder": cache_folder,
              "model_kwargs": {"torch_dtype": torch.float32}}
    if model_name == ENCODER_NAME:
        kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
    encoder = SentenceTransformer(model_name, **kwargs)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def build_title_vectors(graph, encoder=None, dimensions=256, feature_method="semantic"):
    """Encode each title once; no distance labels enter either feature method."""
    nodes = graph["nodes"]
    page_ids = [n["pageid"] for n in nodes]
    if not page_ids or len(set(page_ids)) != len(page_ids):
        raise ValueError("Graph contains no nodes or has duplicate graph node IDs")
    if feature_method == "sha256":
        return {n["pageid"]: title_features(n["title"], dimensions) for n in nodes}
    if feature_method != "semantic":
        raise ValueError("Feature method must be sha256 or semantic")
    if dimensions not in (128, 256, 512, 768):
        raise ValueError("EmbeddingGemma supports 128, 256, 512 or 768 dimensions")
    if encoder is None:
        encoder = create_encoder()
    with torch.inference_mode():
        embeddings = encoder.encode([n["title"] for n in nodes], prompt_name="SentenceSimilarity",
                                    batch_size=32, truncate_dim=dimensions, normalize_embeddings=True,
                                    convert_to_tensor=True, show_progress_bar=True)
    embeddings = torch.as_tensor(embeddings).detach().float().cpu()
    if embeddings.shape != (len(nodes), dimensions) or not torch.isfinite(embeddings).all():
        raise ValueError("Encoder returned an invalid feature matrix")
    # Normalize after truncation, including when a test encoder ignores that option.
    embeddings = nn.functional.normalize(embeddings, dim=1)
    return dict(zip(page_ids, embeddings))


def pair_features(candidate, destination):
    """Same four feature blocks in BOTH methods; no distance/ID inputs."""
    return torch.cat((candidate, destination, (candidate-destination).abs(), candidate*destination), dim=-1)


class DistanceModel(nn.Module):
    def __init__(self, dimensions=128, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimensions*4, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, values):
        return self.network(values).squeeze(-1)
