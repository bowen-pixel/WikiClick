"""Training-only distance-derived targets; inference predicts from embeddings."""
import torch
from torch import nn


class ProgressHead(nn.Linear):
    def __init__(self,hidden=64):
        super().__init__(hidden,1);nn.init.zeros_(self.weight);nn.init.zeros_(self.bias)
    def forward(self,hidden):return torch.sigmoid(super().forward(hidden).squeeze(-1))
