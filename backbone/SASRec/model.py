"""SASRec inference model carried over from the original project.

Derived from pmixer/SASRec.pytorch (Apache-2.0); includes the original project's
legacy and adapted Transformer variants. See LICENSE.txt in this directory.
"""
import math
from torch.nn import functional as F
import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

class LegacySASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(LegacySASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.padding_mask = getattr(args, "padding_mask", False)

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # Keep padding at position 0 and preserve native positions for real items.
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        padding = torch.as_tensor(log_seqs == 0, device=self.dev)
        if self.padding_mask:
            mask = attention_mask[None].expand(seqs.shape[0], -1, -1).clone()
            mask |= padding[:, None, :]
            # Padded queries may attend themselves to prevent all-masked rows/NaNs.
            diagonal = torch.arange(tl, device=self.dev)
            mask[:, diagonal, diagonal] = False
            attention_mask = mask.repeat_interleave(self.attention_layers[0].num_heads, dim=0)
            seqs = seqs.masked_fill(padding[..., None], 0)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

            if self.padding_mask:
                seqs = seqs.masked_fill(padding[..., None], 0)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


class AttentionBlock(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        d = cfg.dim
        self.heads, self.p = cfg.heads, cfg.dropout
        self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.qkv, self.proj = torch.nn.Linear(d, 3*d), torch.nn.Linear(d, d)
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4*d), torch.nn.GELU(), torch.nn.Dropout(self.p), torch.nn.Linear(4*d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = [a.view(b, t, self.heads, d//self.heads).transpose(1, 2) for a in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                          dropout_p=self.p if self.training else 0.0)
        y = self.proj(y.transpose(1, 2).reshape(b, t, d))
        x = x + F.dropout(y, self.p, self.training)
        return x + F.dropout(self.ff(self.ln2(x)), self.p, self.training)


class SASRec(LegacySASRec):
    """Sequence backbone from the supplied model; legacy checkpoints remain loadable."""
    def __init__(self, user_num, item_num, args):
        self.architecture = getattr(args, 'architecture', 'legacy')
        if self.architecture == 'legacy':
            super().__init__(user_num, item_num, args)
            return
        torch.nn.Module.__init__(self)
        self.user_num, self.item_num, self.dev = user_num, item_num, args.device
        self.maxlen, self.dropout = args.maxlen, args.dropout_rate
        self.item_emb = torch.nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.in_norm = torch.nn.LayerNorm(args.hidden_units)
        from types import SimpleNamespace
        cfg = SimpleNamespace(dim=args.hidden_units, heads=args.num_heads, dropout=args.dropout_rate)
        self.blocks = torch.nn.ModuleList([AttentionBlock(cfg) for _ in range(args.num_blocks)])
        self.out_norm = torch.nn.LayerNorm(args.hidden_units)
        self.bias = torch.nn.Parameter(torch.zeros(item_num + 1))
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def log2feats(self, log_seqs):
        if self.architecture == 'legacy':
            return super().log2feats(log_seqs)
        ids = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)
        valid = ids.ne(0)
        t = ids.shape[1]
        positions = torch.arange(self.maxlen - t, self.maxlen, device=ids.device)
        x = self.item_emb(ids) * math.sqrt(self.item_emb.embedding_dim) + self.pos_emb(positions)[None]
        x = F.dropout(self.in_norm(x), self.dropout, self.training) * valid[..., None]
        causal = torch.ones((t, t), device=ids.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        for block in self.blocks:
            x = block(x, allowed) * valid[..., None]
        return self.out_norm(x)

    def catalogue_scores(self, log_seqs):
        scores = self.log2feats(log_seqs)[:, -1] @ self.item_emb.weight[1:].T
        if self.architecture != 'legacy':
            scores = scores + self.bias[1:]
        return scores

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        pos_logits, neg_logits = super().forward(user_ids, log_seqs, pos_seqs, neg_seqs)
        if self.architecture != 'legacy':
            pos_logits = pos_logits + self.bias[torch.as_tensor(pos_seqs, device=self.dev, dtype=torch.long)]
            neg_logits = neg_logits + self.bias[torch.as_tensor(neg_seqs, device=self.dev, dtype=torch.long)]
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        scores = super().predict(user_ids, log_seqs, item_indices)
        if self.architecture != 'legacy':
            scores = scores + self.bias[torch.as_tensor(item_indices, device=self.dev, dtype=torch.long)]
        return scores
