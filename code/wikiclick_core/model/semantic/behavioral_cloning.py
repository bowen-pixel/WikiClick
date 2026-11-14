"""Cosine-residual behavioral-cloning policy; backbone embeddings stay frozen."""
import torch
from torch import nn
from torch.nn import functional as F


class RFBCPolicy(nn.Module):
    def __init__(self,embeddings,page_ids,hidden=64):
        super().__init__()
        if len(page_ids)!=len(embeddings) or len(set(page_ids))!=len(page_ids):raise ValueError('Embedding row map mismatch')
        if embeddings.ndim!=2 or not torch.isfinite(embeddings).all() or (embeddings.norm(dim=1)==0).any():raise ValueError('Invalid frozen embeddings')
        self.register_buffer('embeddings',F.normalize(embeddings.detach().float().cpu(),dim=1),persistent=False)
        self.page_ids=list(page_ids);self.index={n:i for i,n in enumerate(page_ids)}
        dim=embeddings.shape[1]
        self.residual=nn.Sequential(nn.Linear(dim*5,hidden),nn.ReLU(),nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self,rows):
        sizes=[len(r['candidates']) for r in rows]
        candidates=torch.tensor([self.index[n] for r in rows for n in r['candidates']])
        currents=torch.tensor([self.index[r['current']] for r in rows for _ in r['candidates']])
        goals=torch.tensor([self.index[r['goal']] for r in rows for _ in r['candidates']])
        c,v,t=self.embeddings[candidates],self.embeddings[currents],self.embeddings[goals]
        base=(c*t).sum(1)
        scores=base+self.residual(torch.cat([v,t,c,abs(c-t),c*t],1)).squeeze(-1)
        return scores.split(sizes)

    @torch.inference_mode()
    def score(self,current,candidates,target):
        return self([{'current':current,'goal':target,'candidates':candidates}])[0]


class AnchoredPriority:
    """One common score function s(source,u,target) for the whole frontier.

    Current-page-conditioned policy scores are NOT mixed across parents. This
    source-anchored heuristic is a WikiClick search adaptation, not paper RFBC.
    No softmax, distance calibration, or shortest-path guarantee is claimed.
    """
    def __init__(self,policy,source,target):self.policy=policy;self.source=source;self.target=target
    def score(self,current,candidates,target):
        if target!=self.target:raise ValueError('Episode target changed')
        return self.policy.score(self.source,candidates,target)
