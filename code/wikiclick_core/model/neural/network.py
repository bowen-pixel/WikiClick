"""Distance regression head, with compatibility exports for shared features."""
from torch import nn
from ..title_feature.sha256 import title_features
from ..title_feature.embeddinggemma import ENCODER_NAME, create_encoder, build_title_vectors
from ..title_feature.pair import pair_features
FEATURE_VERSION="title-char-3-5-and-words-sha256-v1"
SEMANTIC_FEATURE_VERSION="embeddinggemma-2-title-v1"

class DistanceModel(nn.Module):
    def __init__(self, dimensions=128, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimensions*4, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, values):
        return self.network(values).squeeze(-1)
