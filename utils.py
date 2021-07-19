import torch
import torch.nn as nn
import random
import os
import numpy as np
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from typing import Any, List, Mapping
from torch import Tensor
from torch.optim import Optimizer


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rmse = RMSELoss()
        self.mae = nn.L1Loss()

    def forward(self, yhat: Tensor, y: Tensor) -> Tensor:
        return self.rmse(yhat, y)*0.6 + self.mae(yhat, y)*0.4


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat: Tensor, y: Tensor) -> Tensor:
        return torch.sqrt(self.mse(yhat, y))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def fetch_loss(ltype: str = 'mse') -> Tensor:
    if ltype == 'mse':
        loss = torch.nn.MSELoss()
    elif ltype == 'rmse':
        loss = RMSELoss()
    elif ltype == 'custom':
        loss = CustomLoss()
    return loss


def fetch_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    warm_up: int,
    total_steps: int = None
) -> Any:
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up,
            num_training_steps=total_steps,
        )

    elif scheduler_type == 'cos':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up,
            num_training_steps=total_steps,
        )

    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up
        )

    return scheduler


def get_optimizer_params(learning_rate: float, model: nn.Module, group_type: str = 'a') -> List[Mapping[str, Any]]:
    '''
    Differential LR and Weight Decay
    '''
    no_decay = ['bias', 'gamma', 'beta']

    if group_type == 'a':
        optimizer_parameters = [
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay_rate': 0.01
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay_rate': 0.0},
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if "transformer" not in n
                ],
                'lr': 1e-3,
                'weight_decay_rate':0.01
            }
        ]

    elif group_type == 'b':
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group_all = [
            'layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.',
            'layer.6.', 'layer.7.', 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.'
        ]

        optimizer_parameters = [
            # With Decay
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                'weight_decay': 0.01
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group1)
                ],
                'weight_decay': 0.01,
                'lr': learning_rate/2.6
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group2)
                ]
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group3)
                ],
                'weight_decay': 0.01,
                'lr': learning_rate*2.6
            },
            # Without Decay
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                'weight_decay': 0.0
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group1)
                ],
                'weight_decay': 0.0,
                'lr': learning_rate/2.6
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group2)
                ],
                'weight_decay': 0.0
            },
            {
                'params': [
                    p for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group3)
                ],
                'weight_decay': 0.0,
                'lr': learning_rate*2.6
            },
            # Regressor
            {
                'params': [
                    p for n, p in model.attention.named_parameters()
                ],
                'lr':1e-4,
                'momentum': 0.99
            },
            {
                'params': [
                    p for n, p in model.regressor.named_parameters()
                ],
                'lr':1e-4,
                'momentum': 0.99
            },
        ]

    return optimizer_parameters
