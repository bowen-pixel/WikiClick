"""Pairwise RankNet-style loss and training-only mixed negative sampling."""
import torch
from torch.nn import functional as F

def mixed_negatives(logits, mask, generator, hard=4, random_count=4):
    """Detached hardest negatives plus random remaining negatives, without repeats."""
    negative = torch.nonzero(~mask, as_tuple=False).flatten()
    if not len(negative):
        return negative
    order = torch.argsort(logits.detach()[negative], descending=True, stable=True)
    hardest = negative[order[:hard]]
    rest = negative[order[hard:]]
    sampled = rest[torch.randperm(len(rest), generator=generator)[:random_count]]
    return torch.cat((hardest, sampled))


def ranknet_loss(logits, optimal, generator=None, mixed=False):
    """sigma=1 RankNet logistic loss over all optimal/nonoptimal pairs per decision."""
    mask = torch.as_tensor(optimal, dtype=torch.bool, device=logits.device)
    if mask.shape != logits.shape or not mask.any():
        raise ValueError('One label per candidate and at least one optimal candidate required')
    positive = logits[mask]
    indices = mixed_negatives(logits, mask, generator) if mixed else torch.nonzero(~mask).flatten()
    if not len(indices):
        return logits.sum() * 0  # No preference constraints when every choice is optimal.
    return F.softplus(logits[indices].unsqueeze(0) - positive.unsqueeze(1)).mean()
