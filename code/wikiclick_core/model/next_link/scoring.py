"""Higher scores prefer shorter routes; context adds a learned residual."""
import torch
from torch import nn
from ..neural.network import DistanceModel, pair_features

class Ranker(nn.Module):
    def __init__(self, distance_state, semantic=False):
        super().__init__()
        self.distance = DistanceModel()
        self.distance.load_state_dict(distance_state)
        self.semantic = semantic
        if semantic:
            self.context = nn.Sequential(nn.Linear(512, 32), nn.ReLU(), nn.Linear(32, 1))
            # Begin at the title model; learn additional context during fine-tuning.
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)

    def forward(self, candidates, goals, title, semantic):
        score = -self.distance(pair_features(title[candidates], title[goals]))
        if self.semantic:
            score = score+self.context(pair_features(semantic[candidates], semantic[goals])).squeeze(-1)
        return score


def grouped_batch(rows, title, semantic, model):
    sizes = [len(r["candidates"]) for r in rows]
    candidates = torch.tensor([b for r in rows for b in r["candidates"]])
    goals = torch.tensor([r["goal"] for r in rows for _ in r["candidates"]])
    logits = model(candidates, goals, title, semantic)
    return logits.split(sizes)


def ranking_loss(logits, optimal):
    mask = torch.tensor(optimal, dtype=torch.bool)
    # Uniform targets on all optimal links; an unreachable neighbor is a valid negative.
    return torch.logsumexp(logits, 0)-logits[mask].mean()
