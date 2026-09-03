import torch
import torch.nn as nn
import torch.nn.functional as F

from dataLoader import dataProcess


class SimpleBaseline(nn.Module):
    """
    A lightweight baseline for cascade next-user prediction:
    - User embedding table
    - Causal masked average pooling over the observed cascade
    - Linear projection to user vocabulary

    It matches the DeDiff forward signature: returns (logits, label, process_data).
    """

    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.user_num = args.user_num
        self.embed = nn.Embedding(self.user_num, self.dim, padding_idx=0)
        self.proj = nn.Linear(self.dim, self.user_num)

    def forward(self, args, data, info=None):
        cascade, cas_mask, label, label_mask, neighbor, relation, dis, timestamp = dataProcess(args, data)
        # embeddings: [B, T, E]
        x = self.embed(cascade)

        # build valid mask [B, T, 1] where 1 indicates valid (non-PAD)
        valid = (~cas_mask).float().unsqueeze(-1)

        # causal cumulative sum and count to get mean up to current step
        x_masked = x * valid
        cumsum = torch.cumsum(x_masked, dim=1)
        counts = torch.cumsum(valid, dim=1).clamp_min(1.0)
        h = cumsum / counts

        logits = self.proj(h) + label_mask  # [B, T, V]
        logits = logits.view(-1, logits.size(-1))
        return logits, label, {}

