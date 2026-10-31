"""Attach train-only retrieval and memory to an external backbone at inference."""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import logsumexp
from .config import Config
from .data import validate_histories
from backbone import load_backbone



def fixed_weights(config: Config) -> np.ndarray:
    return np.asarray([1.0, config.retrieval.weight if config.retrieval.enabled else 0.0,
                       config.memory.weight if config.memory.enabled else 0.0], np.float64)


def retrieval_lift(backbone: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Original relative-evidence scale, now with no extra model correction.

    Retrieval itself uses only train cases and the observed query. This score
    conversion compares raw mass with the *unmodified* backbone prior. It does
    not consult item memory or fusion coefficients.
    """
    b, m = np.asarray(backbone,np.float32), np.asarray(mass,np.float32)
    if b.ndim != 2 or m.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(m).all() or np.any(m<0):
        raise ValueError('Expected matching finite real-item scores and nonnegative mass.')
    prior = b - logsumexp(b,axis=1,keepdims=True)
    out = np.zeros_like(b)
    supported = m>0
    out[supported] = np.logaddexp(0.,np.log(m[supported])-prior[supported])
    return out


def fuse(components: np.ndarray, weights) -> np.ndarray:
    """Input [B,N,3] over real items; output [B,N+1], padding excluded."""
    if isinstance(components, torch.Tensor):
        x = components
        w = torch.as_tensor(weights, device=x.device, dtype=torch.float64)
        if x.ndim != 3 or x.shape[2] != 3 or w.shape != (3,) or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError('Expected three nonnegative module coefficients.')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('Nonfinite component scores.')
        values = x[:,:,0].double()*w[0] + x[:,:,1].double()*w[1] + x[:,:,2].double()*w[2]
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite fused scores.')
        padding = torch.full((len(x), 1), -1e9, device=x.device, dtype=torch.float64)
        return torch.cat((padding, values), dim=1)
    x=np.asarray(components);w=np.asarray(weights,np.float64)
    if x.ndim!=3 or x.shape[2]!=3 or w.shape!=(3,) or not np.isfinite(w).all() or np.any(w<0):
        raise ValueError('Expected three nonnegative module coefficients.')
    if not np.isfinite(x).all():raise ValueError('Nonfinite component scores.')
    values=x[:,:,0].astype(np.float64)*w[0]
    values+=x[:,:,1].astype(np.float64)*w[1]
    values+=x[:,:,2].astype(np.float64)*w[2]
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite fused scores.')
    out=np.empty((len(x),x.shape[1]+1),np.float64)
    out[:,0]=-1e9;out[:,1:]=values
    return out


class Recommender:
    """No targets, trees, candidate features or per-user gates in inference."""
    def __init__(self,data,checkpoint,config:Config,*,cache_dir='cache/memory',
                 device='cpu',force_components=False,backbone_config=None,backbone='sasrec',
                 backbone_adapter=None,retrieval_one_case_per_source=True,
                 backbone_source=None):
        self.data,self.config,self.device=data,config,torch.device(device)
        self.backbone=(backbone_adapter if backbone_adapter is not None else
                       load_backbone(backbone,checkpoint,data,self.device,backbone_config,
                                     backbone_source))
        self.weights=fixed_weights(config)
        self.retriever=None;self.memory=None
        if config.retrieval.enabled and (force_components or self.weights[1]>0):
            from .retrieval import Retriever
            self.retriever=Retriever(data.raw['train'],data.num_items,config.retrieval,
                                     device=self.device,
                                     one_case_per_source=retrieval_one_case_per_source)
        if config.memory.enabled and (force_components or self.weights[2]>0):
            from .memory import ItemMemory
            self.memory=ItemMemory.fit(data.raw['train'],data.num_items,cache_dir,device=self.device)

    @torch.inference_mode()
    def components(self,histories,query_users,*,return_tensor=False):
        hs=validate_histories(histories,self.data.num_items)
        users=np.asarray(query_users)
        if users.shape!=(len(hs),) or users.dtype.kind not in 'iu' or np.any(users<0) or np.any(users>np.iinfo(np.int64).max):
            raise ValueError('Exactly one nonnegative user ID per query is required.')
        if len(hs)>self.config.inference_batch_size:raise ValueError('Split queries at inference_batch_size.')
        scores=self.backbone.score(hs)
        if (not isinstance(scores, torch.Tensor)
                or scores.shape != (len(hs), self.data.num_items)
                or scores.is_complex()
                or not bool(torch.isfinite(scores).all())):
            raise ValueError('Backbone must return finite scores with shape [batch, num_items].')
        scores=scores.float()
        if self.device.type == 'cuda':
            x=torch.zeros((len(hs),self.data.num_items,3),device=self.device)
            x[:,:,0]=scores
            if self.retriever is not None:
                mass=self.retriever.query(hs,users)[:,1:]
                prior=torch.log_softmax(scores,dim=1)
                supported=mass>0
                lift=torch.zeros_like(scores)
                lift[supported]=torch.logaddexp(torch.zeros_like(mass[supported]),
                                                 torch.log(mass[supported])-prior[supported])
                x[:,:,1]=lift
            if self.memory is not None:
                x[:,:,2]=self.memory.score(hs)
            return x if return_tensor else x.cpu().numpy()
        scores=scores.cpu().numpy()
        x=np.zeros((len(hs),self.data.num_items,3),np.float32);x[:,:,0]=scores
        if self.retriever is not None:
            mass=self.retriever.query(hs,users)[:,1:]
            x[:,:,1]=retrieval_lift(scores,mass)
        if self.memory is not None:x[:,:,2]=self.memory.score(hs)
        return torch.as_tensor(x, device=self.device) if return_tensor else x

    def predict(self,histories,query_users):
        return fuse(self.components(histories,query_users,return_tensor=self.device.type == 'cuda'),self.weights)
