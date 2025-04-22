# moe_layer.py — Mixture of Experts Layer (Grok‑3 Style)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_scores = self.gate(x)  # Shape: [batch_size, num_experts]
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Normalize topk scores (softmax over top_k)
        topk_weights = F.softmax(topk_scores, dim=-1)

        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(1)

            for b in range(x.size(0)):
                expert_out = self.experts[expert_idx[b]](x[b].unsqueeze(0))
                output[b] += weight[b] * expert_out.squeeze(0)

        return output
