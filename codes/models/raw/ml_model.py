"""Small title-feature model: page IDs are identities, never learned inputs."""

import hashlib
import re

import torch
from torch import nn

FEATURE_VERSION = "title-char-3-5-and-words-sha256-v1"


def title_features(title, dimensions=128):
    """Fixed hashing works for unseen titles without fitting a test vocabulary."""
    text = " " + " ".join(title.lower().split()) + " "
    tokens = re.findall(r"\w+", text)
    tokens += [text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1))]
    result = torch.zeros(dimensions)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "little") % dimensions] += 1 if digest[4] % 2 else -1
    return result / result.norm().clamp_min(1)


def pair_features(candidate, destination):
    """Concatenate both titles and two simple comparisons; no distance inputs."""
    return torch.cat((candidate, destination, (candidate-destination).abs(), candidate*destination), dim=-1)


class DistanceModel(nn.Module):
    def __init__(self, dimensions=128, hidden=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimensions*4, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, values):
        return self.network(values).squeeze(-1)
