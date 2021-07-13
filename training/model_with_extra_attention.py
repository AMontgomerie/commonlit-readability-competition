"""Based on https://www.kaggle.com/chamecall/clrp-finetune"""

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, in_size=768, hidden_size=512):
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class CLRPModel(nn.Module):
    def __init__(self, transformer, hidden_size=512):
        super(CLRPModel, self).__init__()
        self.transformer = transformer
        self.head = AttentionHead(
            in_size=transformer.config.hidden_size,
            hidden_size=hidden_size
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer(input_ids, attention_mask)
        x = self.head(transformer_out.last_hidden_state)
        x = self.linear(x)
        return {"logits": x}
