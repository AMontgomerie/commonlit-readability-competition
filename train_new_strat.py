from types import SimpleNamespace
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from typing import List, Tuple
from dataclasses import dataclass

from utils import seed_everything, fetch_loss, get_optimizer_params, fetch_scheduler
from data_prep import make_loaders
from models import TransformerWithAttentionHead
from training import Trainer


@dataclass()
class Config:
    device: torch.device = torch.device('cuda')
    max_length: int = 248
    eval_schedule: List[Tuple[float, int]] = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]
    print_step: int = 10
    num_workers: int = 4
    train_batch_size: int = 4
    valid_batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    loss_type: str = "mse"
    use_diff_lr: bool = True
    model_checkpoint: str = "microsoft/deberta-large"
    scheduler: str = "cos"
    seed: int = 1000


def train_cv(data: pd.DataFrame, folds: List[int], config: Config) -> None:
    loss_cv = []

    for fold in folds:
        print(f'Starting for fold {fold} and seed {config.seed+fold} \n')
        seed_everything(config.seed+fold)

        tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
        train_loader, valid_loader = make_loaders(
            data,
            tokenizer,
            fold,
            train_bs=config.train_batch_size,
            valid_bs=config.valid_batch_size,
            num_workers=config.num_workers
        )
        model = TransformerWithAttentionHead(config.model_checkpoint)
        model.to(config.device)
        criterion = fetch_loss(ltype=config.loss_type)
        criterion.to(config.device)

        if config.use_diff_lr:
            print('yes')
            optimizer_parameters = get_optimizer_params(model, group_type='b')

        else:
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {
                    'params': [
                        p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    'weight_decay': config.weight_decay
                },
                {
                    'params': [
                        p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    'weight_decay': 0.0
                },
            ]

        optimizer = AdamW(
            optimizer_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        total_steps = len(train_loader)*config.epochs
        scheduler = fetch_scheduler(optimizer, 50, total_steps)

        trainer = Trainer(model, optimizer, scheduler, criterion, seed=fold+config.seed)

        best_score = 100000
        for epoch in range(config.epochs):
            print("\n\n")
            print(f"###### BEGINNING EPOCH {epoch+1} ##########")
            best_score = trainer.train(
                train_loader,
                valid_loader,
                best_score,
                config.device,
                epoch,
                fold
            )
            print(f'Valid Score at epoch {epoch+1} end : {best_score}')

        loss_cv.append(best_score)

    print('RMSE by fold:')
    for fold, result in enumerate(loss_cv):
        print(f"{fold}: {result}")

    print('CV RMSE:', np.mean(loss_cv))


if __name__ == "__main__":
    data = pd.read_csv('data/train_folds.csv')
    folds = list(data.kfold.unique())
    config = Config()
    train_cv(data, folds, config)
