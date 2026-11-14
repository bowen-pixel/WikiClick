"""Shared local scoring interface. Cosine also supplies comparable node priorities."""
from typing import Protocol
import torch
from torch.nn import functional as F


class NavigationPolicy(Protocol):
    def score(self,current:int,candidates:list[int],target:int)->torch.Tensor: ...


class CosinePolicy:
    def __init__(self,embeddings,page_ids):
        if embeddings.ndim!=2 or len(embeddings)!=len(page_ids) or len(set(page_ids))!=len(page_ids):
            raise ValueError('Embedding rows and unique page IDs must align')
        if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=1)==0).any():
            raise ValueError('Expected finite nonzero document embeddings')
        self.embeddings=F.normalize(embeddings.detach().float().cpu(),dim=1)
        self.index={n:i for i,n in enumerate(page_ids)}

    @torch.inference_mode()
    def score(self,current,candidates,target):
        if current not in self.index:raise ValueError('Unknown current article')
        indices=torch.tensor([self.index[n] for n in candidates],dtype=torch.long)
        return self.embeddings[indices]@self.embeddings[self.index[target]]
