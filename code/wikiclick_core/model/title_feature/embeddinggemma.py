"""Frozen optional pretrained title vectors."""
import torch
from torch import nn
from .sha256 import title_features
ENCODER_NAME="google/embeddinggemma-2"

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
