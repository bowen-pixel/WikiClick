"""Parent-independent residual priority and optional shared progress auxiliary."""
import torch
from torch import nn
from torch.nn import functional as F
from .progress_head import ProgressHead


class ResidualRanker(nn.Module):
    def __init__(self,dimension=128):
        super().__init__();self.dimension=dimension
        self.trunk=nn.Sequential(nn.Linear(dimension*4,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
        self.correction=nn.Linear(64,1);nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)
        self.progress=ProgressHead(64)
    def forward(self,candidate,target):
        hidden=self.trunk(torch.cat([candidate,target,abs(candidate-target),candidate*target],dim=-1))
        return self.correction(hidden).squeeze(-1),self.progress(hidden)


def losses(model,embeddings,batch,model_kind,regularization,beta,hard_weight=0.):
    c=embeddings[batch['candidate']];t=embeddings[batch['target']]
    correction,q=model(c,t);scores=correction+(c*t).sum(1) if model_kind=='M2' else correction
    owner=batch['owner'];count=batch['size'];positive=batch['positive']
    maximum=torch.full((count,),-torch.inf,dtype=scores.dtype).scatter_reduce(0,owner,scores,reduce='amax',include_self=True)
    exponent=(scores-maximum[owner]).exp();denom=torch.zeros(count).scatter_add(0,owner,exponent)
    numerator=torch.zeros(count).scatter_add(0,owner,exponent*positive)
    ranking=(denom.clamp_min(1e-30).log()-numerator.clamp_min(1e-30).log()).mean()
    progress=F.mse_loss(q,batch['progress']);penalty=correction.square().mean()
    hard=scores.sum()*0
    if hard_weight and len(batch.get('hard_indices',[])):
        best=torch.full((count,),-torch.inf).scatter_reduce(0,owner,scores.masked_fill(~positive,-torch.inf),reduce='amax',include_self=True)
        ids=batch['hard_indices'];values=F.softplus(scores[ids]-best[owner[ids]])
        sums=torch.zeros(count).scatter_add(0,owner[ids],values);counts=torch.zeros(count).scatter_add(0,owner[ids],torch.ones_like(values))
        hard=(sums/counts.clamp_min(1)).mean()
    total=ranking+beta*progress+regularization*penalty+hard_weight*hard
    return total,{'ranking':float(ranking.detach()),'progress_mse':float(progress.detach()),'correction_square':float(penalty.detach()),'hard_pair':float(hard.detach()),'total':float(total.detach())}
