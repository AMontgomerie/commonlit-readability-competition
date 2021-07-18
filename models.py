"""Based on https://www.kaggle.com/chamecall/clrp-finetune"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel
from types import SimpleNamespace


class AttentionHead(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, features: Tensor) -> Tensor:
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class TransformerWithAttentionHead(nn.Module):
    def __init__(
        self,
        transformer_checkpoint: str,
        attn_hidden_size: int = 768,
        hidden_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-7
    ) -> None:
        super(TransformerWithAttentionHead, self).__init__()
        config = AutoConfig.from_pretrained(transformer_checkpoint)
        config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": hidden_dropout_prob,
            "layer_norm_eps": layer_norm_eps
        })
        self.transformer = AutoModel.from_pretrained(transformer_checkpoint, config=config)
        self.attn_head = AttentionHead(
            in_size=config.hidden_size,
            hidden_size=attn_hidden_size
        )
        self.regressor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> SimpleNamespace:
        transformer_out = self.transformer(input_ids, attention_mask)
        x = self.attn_head(transformer_out.last_hidden_state)
        x = self.regressor(x)
        return SimpleNamespace(logits=x)