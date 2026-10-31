"""Exact train-only case retrieval, independent of learned evidence strength.

One best case per source user, query-user exclusion, deterministic score/ID
ordering. CSR construction and source reduction avoid huge Python intermediates.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch
from scipy import sparse
from numba import njit
from .config import RetrievalConfig
from .data import validate_histories
from .utils import training_digest
from .progress import progress, stage

GPU_SCORE_ELEMENTS = 4_000_000


@njit(cache=True, nogil=True)
def _fill_cases(sequences, offsets, users, idf, max_len, horizon,
                indptr, indices, values, source, groups, outcomes, weights):
    row=0;ptr=0
    for u in range(len(users)):
        start=offsets[u];length=offsets[u+1]-start
        for end in range(1,length):
            source[row]=users[u];groups[row]=u;indptr[row]=ptr
            first=max(0,end-max_len)
            for t in range(first,end):
                item=sequences[start+t];indices[ptr]=item
                w=idf[item]
                values[ptr]=w;ptr+=1
            count=min(horizon,length-end)
            for t in range(count):
                outcomes[row,t]=sequences[start+end+t];weights[row,t]=np.float32(1.0/count)
            row+=1
    indptr[row]=ptr


@njit(cache=True)
def _best_sources(ids,scores,groups,users,query_user):
    best=np.zeros(len(users),np.float32);cases=np.full(len(users),-1,np.int64)
    touched=np.empty(len(users),np.int64);n=0
    for j in range(len(ids)):
        c=ids[j];u=groups[c];s=scores[j]
        if users[u]==query_user or s<=0:continue
        if cases[u]<0:
            touched[n]=u;n+=1;best[u]=s;cases[u]=c
        elif s>best[u] or (s==best[u] and c<cases[u]):
            best[u]=s;cases[u]=c
    result=np.empty(n,np.int64);ss=np.empty(n,np.float32)
    for j in range(n):
        u=touched[j];result[j]=cases[u];ss[j]=best[u]
    return result,ss


@dataclass
class Retrieved:
    cases: np.ndarray
    similarities: np.ndarray
    query_users: np.ndarray
    memory_digest: str


class Retriever:
    def __init__(self,train:dict,num_items:int,config:RetrievalConfig,device='cpu',*,one_case_per_source=True):
        if not isinstance(train,dict) or not train or not isinstance(config,RetrievalConfig):raise ValueError('Invalid training memory.')
        if type(num_items) is not int or num_items<1:raise ValueError('num_items must be a positive integer.')
        if type(one_case_per_source) is not bool:raise ValueError('one_case_per_source must be boolean.')
        self.n=num_items;self.config=config;self.device=torch.device(device)
        self.one_case_per_source=one_case_per_source
        if any(type(u) is not int or not 0 <= u < 2**63 for u in train):
            raise ValueError('Source user IDs must be nonnegative int64 integers.')
        users=np.asarray(sorted(train),np.int64);histories=validate_histories([train[int(u)] for u in users],num_items)
        lengths=np.asarray([len(h) for h in histories],np.int64)
        if np.any(lengths<2):raise ValueError('Source histories require at least two events.')
        df=np.zeros(num_items+1,np.int64)
        for h in progress(histories,desc='[retrieval] Item frequencies',unit='user'):df[np.unique(h)]+=1
        self.idf=(np.log((1.+len(users))/(1.+df))+1.).astype(np.float32);self.idf[0]=0
        count=int(np.sum(lengths-1));limit=config.max_len
        # Sum min(end,L) for end=1..length-1.
        last=lengths-1;m=np.minimum(last,limit)
        nnz=int(np.sum(m*(m+1)//2+np.maximum(last-limit,0)*limit))
        indptr=np.empty(count+1,np.int64);indices=np.empty(nnz,np.int32);values=np.empty(nnz,np.float32)
        self.sources=np.empty(count,np.int64);self.groups=np.empty(count,np.int32);self.users=users
        self.outcome_ids=np.zeros((count,config.horizon),np.int32)
        self.outcome_weights=np.zeros((count,config.horizon),np.float32)
        with stage('[retrieval] Building retrieval cases'):
            _fill_cases(np.concatenate(histories),np.r_[0,np.cumsum(lengths)],users,self.idf,limit,config.horizon,
                        indptr,indices,values,self.sources,self.groups,self.outcome_ids,self.outcome_weights)
        with stage('[retrieval] Building sparse index'):
            x=sparse.csr_matrix((values,indices,indptr),shape=(count,num_items+1));x.sum_duplicates();x.sort_indices()
            norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
            self.matrix=(sparse.diags(1/np.maximum(norm,1e-12))@x).tocsr().astype(np.float32)
            self.matrix.sort_indices()
            self.transpose=self.matrix.T.tocsr()
        if self.device.type == 'cuda':
            csr=self.matrix
            self._crow=torch.from_numpy(csr.indptr.astype(np.int64)).to(self.device)
            self._col=torch.from_numpy(csr.indices.astype(np.int64)).to(self.device)
            self._value=torch.from_numpy(csr.data.copy()).to(self.device)
            self._groups=torch.from_numpy(self.groups.astype(np.int64)).to(self.device)
            self._users=torch.from_numpy(self.users).to(self.device)
            self._outcome_ids=torch.from_numpy(self.outcome_ids.astype(np.int64)).to(self.device)
            self._outcome_weights=torch.from_numpy(self.outcome_weights).to(self.device)
            torch.cuda.synchronize(self.device)
        self.digest=hashlib.sha256(json.dumps({'training':training_digest(train,num_items),'config':config.index_dict(),
                                              'one_case_per_source':one_case_per_source},sort_keys=True).encode()).hexdigest()
        print(f'[retrieval] {count:,} cases / {len(users):,} sources / {self.matrix.nnz:,} nonzeros',flush=True)

    def _features(self,histories):
        rows=[];cols=[];values=[]
        for row,history in enumerate(histories):
            h=history[-self.config.max_len:];w=self.idf[h]
            rows.extend([row]*len(h));cols.extend(h.tolist());values.extend(w.tolist())
        x=sparse.csr_matrix((np.asarray(values,np.float32),(rows,cols)),shape=(len(histories),self.n+1),dtype=np.float32)
        norm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
        return (sparse.diags(1/np.maximum(norm,1e-12),format='csr')@x).astype(np.float32)

    def retrieve(self,histories,query_users):
        hs=validate_histories(histories,self.n);users=np.asarray(query_users)
        if users.shape!=(len(hs),) or users.dtype.kind not in 'iu' or np.any(users<0) or np.any(users>np.iinfo(np.int64).max):raise ValueError('One nonnegative integer user per query is required.')
        if self.device.type == 'cuda':
            return self._retrieve_gpu(hs,users.astype(np.int64))
        k=self.config.neighbors;cases=np.full((len(hs),k),-1,np.int64);sim=np.zeros((len(hs),k),np.float32)
        for lo in range(0,len(hs),16):
            product=(self._features(hs[lo:lo+16])@self.transpose).tocsr()
            for j in range(product.shape[0]):
                start,end=product.indptr[j:j+2]
                if self.one_case_per_source:
                    ids,ss=_best_sources(product.indices[start:end],product.data[start:end],self.groups,self.users,int(users[lo+j]))
                else:
                    ids,ss=product.indices[start:end],product.data[start:end]
                    keep=(ss>0)&(self.sources[ids]!=users[lo+j])
                    ids,ss=ids[keep],ss[keep]
                if len(ss)>k:
                    threshold=np.partition(ss,len(ss)-k)[len(ss)-k]
                    keep=ss>=threshold;ids=ids[keep];ss=ss[keep]
                order=np.lexsort((ids,-ss))[:k]
                cases[lo+j,:len(order)]=ids[order];sim[lo+j,:len(order)]=ss[order]
        return Retrieved(cases,sim,users.astype(np.int64),self.digest)

    @torch.inference_mode()
    def _retrieve_gpu(self,histories,users):
        """Score bounded case chunks on CUDA and reduce to one case per source."""
        if not self.one_case_per_source:
            return self._retrieve_gpu_unrestricted(histories,users)
        k=self.config.neighbors
        result_cases=[];result_scores=[]
        for lo in range(0,len(histories),16):
            batch=histories[lo:lo+16]
            q=torch.from_numpy(self._features(batch).toarray()).to(self.device).T.contiguous()
            count=len(self.sources);n_sources=len(self.users)
            best=torch.full((len(batch),n_sources),-torch.inf,device=self.device)
            best_case=torch.full((len(batch),n_sources),count,device=self.device,dtype=torch.long)
            chunk=max(1,min(count,GPU_SCORE_ELEMENTS//len(batch)))
            for start in range(0,count,chunk):
                end=min(start+chunk,count)
                first=int(self.matrix.indptr[start]);last=int(self.matrix.indptr[end])
                index=torch.sparse_csr_tensor(self._crow[start:end+1]-first,
                    self._col[first:last],self._value[first:last],
                    size=(end-start,self.n+1),device=self.device)
                scores=torch.sparse.mm(index,q).T.contiguous()
                slots=self._groups[start:end].expand(len(batch),-1)
                chunk_best=torch.full_like(best,-torch.inf)
                chunk_best.scatter_reduce_(1,slots,scores,reduce='amax',include_self=True)
                ids=torch.arange(start,end,device=self.device).expand(len(batch),-1)
                matching=(scores>0)&(scores==chunk_best.gather(1,slots))
                chunk_case=torch.full_like(best_case,count)
                chunk_case.scatter_reduce_(1,slots,torch.where(matching,ids,count),reduce='amin',include_self=True)
                better=chunk_best>best
                tied=chunk_best==best
                best_case=torch.where(better,chunk_case,torch.where(tied,torch.minimum(best_case,chunk_case),best_case))
                best=torch.maximum(best,chunk_best)
            own=torch.searchsorted(self._users,torch.from_numpy(users[lo:lo+len(batch)]).to(self.device))
            own_valid=own<n_sources
            own_valid &= self._users[own.clamp_max(n_sources-1)]==torch.from_numpy(users[lo:lo+len(batch)]).to(self.device)
            rows=torch.arange(len(batch),device=self.device)[own_valid]
            best[rows,own[own_valid]]=-torch.inf
            best_case[rows,own[own_valid]]=count
            best=torch.where(best_case<count,best,torch.full_like(best,-torch.inf))
            by_case=torch.argsort(best_case,dim=1,stable=True)
            by_score=torch.argsort(best.gather(1,by_case),dim=1,descending=True,stable=True)
            slots=by_case.gather(1,by_score)[:,:min(k,n_sources)]
            selected=best_case.gather(1,slots);similarity=best.gather(1,slots)
            valid=torch.isfinite(similarity)
            out_case=torch.full((len(batch),k),-1,dtype=torch.long,device=self.device)
            out_score=torch.zeros((len(batch),k),dtype=torch.float32,device=self.device)
            out_case[:,:len(slots[0])]=torch.where(valid,selected,-1)
            out_score[:,:len(slots[0])]=torch.where(valid,similarity,0.)
            result_cases.append(out_case);result_scores.append(out_score)
        return Retrieved(torch.cat(result_cases),torch.cat(result_scores),
                         torch.from_numpy(users).to(self.device),self.digest)

    @torch.inference_mode()
    def _retrieve_gpu_unrestricted(self,histories,users):
        """Exact top-K cases with stable score/case-ID ordering and no self cases."""
        k=self.config.neighbors
        count=len(self.sources)
        if count>=2**32:
            raise ValueError('GPU case IDs must fit in 32 bits.')
        result_cases=[];result_scores=[]
        for lo in range(0,len(histories),16):
            batch=histories[lo:lo+16]
            query=torch.from_numpy(self._features(batch).toarray()).to(self.device).T.contiguous()
            query_users=torch.from_numpy(users[lo:lo+len(batch)]).to(self.device)
            best=torch.empty((len(batch),0),device=self.device,dtype=torch.int64)
            chunk=max(1,min(count,GPU_SCORE_ELEMENTS//len(batch)))
            for start in range(0,count,chunk):
                end=min(start+chunk,count)
                first=int(self.matrix.indptr[start]);last=int(self.matrix.indptr[end])
                index=torch.sparse_csr_tensor(self._crow[start:end+1]-first,
                    self._col[first:last],self._value[first:last],
                    size=(end-start,self.n+1),device=self.device)
                scores=torch.sparse.mm(index,query).T.contiguous()
                ids=torch.arange(start,end,device=self.device,dtype=torch.int64)[None,:]
                # Positive float32 bit patterns sort exactly like their values.
                # The low bits prefer smaller case IDs when similarities tie.
                keys=(scores.view(torch.int32).to(torch.int64)<<32)|(0xffffffff-ids)
                own=self._users[self._groups[start:end]]==query_users[:,None]
                keys=torch.where((scores>0)&~own,keys,-1)
                chunk_best=torch.topk(keys,min(k,end-start),dim=1).values
                best=torch.topk(torch.cat((best,chunk_best),dim=1),
                                min(k,best.shape[1]+chunk_best.shape[1]),dim=1).values
            valid=best>=0
            case=torch.where(valid,0xffffffff-(best&0xffffffff),-1)
            similarity=torch.where(valid,(best>>32).to(torch.int32).view(torch.float32),0.)
            if best.shape[1]<k:
                case=torch.nn.functional.pad(case,(0,k-best.shape[1]),value=-1)
                similarity=torch.nn.functional.pad(similarity,(0,k-best.shape[1]))
            result_cases.append(case)
            result_scores.append(similarity)
        return Retrieved(torch.cat(result_cases),torch.cat(result_scores),
                         torch.from_numpy(users).to(self.device),self.digest)

    def evidence(self,retrieved):
        if not isinstance(retrieved,Retrieved) or retrieved.memory_digest!=self.digest:raise ValueError('Mismatched retrieval result.')
        if isinstance(retrieved.cases,torch.Tensor):
            return self._evidence_gpu(retrieved)
        cases,sim=retrieved.cases,retrieved.similarities
        if cases.ndim!=2 or cases.shape!=sim.shape or not np.isfinite(sim).all() or np.any(sim<0) or np.any((cases< -1)|(cases>=len(self.sources))):raise ValueError('Invalid retrieval arrays.')
        valid=cases>=0;ix=cases.clip(0)
        for row in range(len(cases)):
            u=self.sources[ix[row,valid[row]]]
            if np.any(u==retrieved.query_users[row]):raise ValueError('Source leakage.')
            if self.one_case_per_source and len(np.unique(u))!=len(u):raise ValueError('Repeated source.')
        ids=self.outcome_ids[ix];weight=(sim[:,:,None]*self.outcome_weights[ix]*valid[:,:,None]).astype(np.float32)
        rows=np.broadcast_to(np.arange(len(cases))[:,None,None],ids.shape)
        out=np.zeros((len(cases),self.n+1),np.float32);np.add.at(out,(rows.ravel(),ids.ravel()),weight.ravel());out[:,0]=0
        return out

    @torch.inference_mode()
    def _evidence_gpu(self,retrieved):
        cases,sim=retrieved.cases,retrieved.similarities
        if cases.ndim!=2 or cases.shape!=sim.shape or cases.dtype!=torch.long or cases.device.type!=self.device.type:
            raise ValueError('Invalid CUDA retrieval arrays.')
        if bool(((cases< -1)|(cases>=len(self.sources))).any()) or not bool(torch.isfinite(sim).all()) or bool((sim<0).any()):
            raise ValueError('Invalid CUDA retrieval arrays.')
        valid=cases>=0;ix=cases.clamp_min(0)
        groups=self._groups[ix]
        source=self._users[groups]
        if bool((valid & (source==retrieved.query_users[:,None])).any()):
            raise ValueError('Source leakage.')
        if self.one_case_per_source:
            for column in range(cases.shape[1]):
                if bool((valid[:,column,None] & valid[:,:column] & (groups[:,column,None]==groups[:,:column])).any()):
                    raise ValueError('Repeated source.')
        ids=self._outcome_ids[ix]
        weights=sim[:,:,None]*self._outcome_weights[ix]*valid[:,:,None]
        out=torch.zeros((len(cases),self.n+1),device=self.device)
        out.scatter_add_(1,ids.reshape(len(cases),-1),weights.reshape(len(cases),-1))
        out[:,0]=0
        return out

    def query(self,histories,query_users):return self.evidence(self.retrieve(histories,query_users))
