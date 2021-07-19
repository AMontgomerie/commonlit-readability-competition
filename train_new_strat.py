import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler
from typing import List
import gc

from utils import seed_everything, fetch_loss, get_optimizer_params
from data_prep import make_loaders
from models import TransformerWithAttentionHead
from training import Trainer, Config
from commonlit_logging import generate_oof_and_log

DEFAULT_EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]


def train_cv(data: pd.DataFrame, folds: List[int], config: Config) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
    loss_cv = []

    for fold in folds:
        rmse = train_fold(data, tokenizer, fold, config)
        loss_cv.append(rmse)

    print('RMSE by fold:')

    for fold, result in enumerate(loss_cv):
        print(f"{fold}: {result}")

    print('CV RMSE:', np.mean(loss_cv))

    generate_oof_and_log(data, tokenizer, config, loss_cv)


def train_fold(data: pd.DataFrame, tokenizer: AutoTokenizer, fold: int, config: Config) -> None:
    print(f"Starting for fold {fold} and seed {config.seed+fold}")
    seed_everything(config.seed+fold)

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
        optimizer_parameters = get_optimizer_params(
            config.learning_rate,
            model=model,
            group_type='b'
        )

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
                'weight_decay': config.weight_decay
            },
        ]

    optimizer = AdamW(
        optimizer_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader)*config.epochs
    scheduler = get_scheduler(config.scheduler, optimizer, config.warmup, total_steps)

    trainer = Trainer(
        config.model_name,
        model,
        optimizer,
        scheduler,
        criterion,
        seed=fold+config.seed,
        save_path=config.save_path,
        print_step=config.print_step,
        eval_schedule=config.eval_schedule
    )

    best_score = 1.
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

    del model
    gc.collect()

    return best_score


if __name__ == "__main__":
    data = pd.read_csv('data/train_folds.csv')
    folds = list(data.kfold.unique())
    config = Config(eval_schedule=DEFAULT_EVAL_SCHEDULE)
    train_cv(data, folds, config)
