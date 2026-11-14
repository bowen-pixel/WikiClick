"""Lightweight paper-inspired contextual ranking; no changes to shared Ranker."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(dimensions, hidden=64):
    return nn.Sequential(nn.Linear(dimensions, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))


def zero_last(network):
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)


class SemanticRanker(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.base = mlp(1024)
        if level >= 2:
            self.current = mlp(768, 32)
            zero_last(self.current)
        if level >= 3:
            self.query = nn.Linear(512, 32)
            self.key = nn.Linear(256, 32)
            self.value = nn.Linear(256, 32)
            self.attention_score = mlp(128, 32)
            zero_last(self.attention_score)
        if level >= 4:
            self.local_score = mlp(6, 16)
            zero_last(self.local_score)

    def forward(self, rows, features):
        sizes = [len(r['candidates']) for r in rows]
        candidates = torch.tensor([n for r in rows for n in r['candidates']])
        goals = torch.tensor([r['goal'] for r in rows for _ in r['candidates']])
        current = torch.tensor([r['current'] for r in rows for _ in r['candidates']])
        z = features['hashes'] if self.level == 0 else features['z']
        c, t, v = z[candidates], z[goals], z[current]
        score = self.base(torch.cat([c, t, abs(c-t), c*t], 1)).squeeze(-1)
        if self.level >= 2:
            score = score+self.current(torch.cat([v, c, t], 1)).squeeze(-1)
        if self.level >= 3:
            q, k, value = self.query(torch.cat([v, t], 1)), self.key(c), self.value(c)
            summaries = []
            offset = 0
            for count in sizes:
                weights = ((q[offset:offset+count]*k[offset:offset+count]).sum(1)/math.sqrt(32)).softmax(0)
                summary = (weights[:, None]*value[offset:offset+count]).sum(0)
                summaries.append(summary.expand(count, -1))
                offset += count
            summary = torch.cat(summaries)
            score = score+self.attention_score(torch.cat([k, q, summary, k*summary], 1)).squeeze(-1)
        if self.level >= 4:
            p, mask = features['passages'][candidates], features['passage_mask'][candidates]
            a = (p*features['title'][goals, None, :]).sum(-1)
            b = (p*features['intro'][goals, None, :]).sum(-1)
            def pooling(x):
                maximum = x.masked_fill(~mask, -1e4).amax(1)
                maximum = torch.where(mask.any(1), maximum, torch.zeros_like(maximum))
                mean = (x*mask).sum(1)/mask.sum(1).clamp_min(1)
                return maximum, mean
            am, aa = pooling(a)
            bm, ba = pooling(b)
            global_cos = (features['intro'][candidates]*features['intro'][goals]).sum(1)
            local = torch.stack([am, aa, bm, ba, global_cos, mask.any(1).float()], 1)
            score = score+self.local_score(local).squeeze(-1)
        return score.split(sizes)


class AuxiliaryRanker(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.auxiliary = mlp(7, 16)
        zero_last(self.auxiliary)

    def forward(self, rows, features, remove_auxiliary=False):
        base = self.base(rows, features)
        if remove_auxiliary:
            return base
        sizes = [len(r['candidates']) for r in rows]
        c = torch.tensor([n for r in rows for n in r['candidates']])
        t = torch.tensor([r['goal'] for r in rows for _ in r['candidates']])
        v = torch.tensor([r['current'] for r in rows for _ in r['candidates']])
        category = features['category']
        aux = torch.cat([features['degree'][c], features['degree'][v],
            (category[c]*category[t]).sum(1, keepdim=True),
            (category[v]*category[t]).sum(1, keepdim=True), torch.ones(len(c), 1)], 1)
        return tuple(x+y for x, y in zip(base, self.auxiliary(aux).squeeze(-1).split(sizes)))
