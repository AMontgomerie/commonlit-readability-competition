import os
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer
from typing import List
from sklearn.metrics import mean_squared_error

from utils import seed_everything, fetch_loss, get_optimizer_params, fetch_scheduler
from data_prep import make_loaders
from models import TransformerWithAttentionHead
from training import Trainer, Config
from log import make_oofs, save_log

DEFAULT_EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]


def train_cv(data: pd.DataFrame, folds: List[int], config: Config) -> None:
    loss_cv = []
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    for fold in folds:
        print(f'Starting for fold {fold} and seed {config.seed+fold} \n')
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
            print('yes')
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

    oof_pred, oof_tar, oof_id, oof_fold = make_oofs(data, tokenizer, config)
    oof = np.concatenate(oof_pred)
    true = np.concatenate(oof_tar)
    id = np.concatenate(oof_id)
    folds = np.concatenate(oof_fold)
    oof_rmse = mean_squared_error(true, oof, squared=False)
    print('Overall OOF RMSE = %.3f' % oof_rmse)

    df_oof = pd.DataFrame({
        "id": id, "target": true, "pred": oof, "fold": folds
    })
    df_oof.to_csv(os.path.join(config.save_path, "roberta_oof.csv"), index=False)

    save_log(config.save_path, loss_cv, oof_rmse, config)


if __name__ == "__main__":
    data = pd.read_csv('data/train_folds.csv')
    folds = list(data.kfold.unique())
    config = Config(eval_schedule=DEFAULT_EVAL_SCHEDULE)
    train_cv(data, folds, config)
