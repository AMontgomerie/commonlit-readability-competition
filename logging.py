import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import List, Tuple
import tqdm

from training import Config
from models import TransformerWithAttentionHead
from utils import seed_everything
from data_prep import ReadabilityDataset


@torch.no_grad()
def oof_out(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> Tuple[List, List, List]:
    model.eval()
    fin_out = []
    fin_tar = []
    fin_id = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        targets = batch[2].to(device)
        ids = batch[3]

        output = model(input_ids, attention_mask)
        output = output.squeeze(1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        fin_out.append(output)
        fin_tar.append(targets)
        fin_id.append(ids)

    return np.concatenate(fin_out), np.concatenate(fin_tar), np.concatenate(fin_id)


def make_oofs(
    data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    config: Config
) -> Tuple[List, List, List, List]:
    oof_id = []
    oof_fold = []
    oof_pred = []
    oof_tar = []

    model = TransformerWithAttentionHead(config.model_checkpoint)
    model.to(config.device)
    model.eval()

    for fold in range(5):
        seed_everything(config.seed+fold)

        valid = data[data['kfold'] == fold].reset_index(drop=True)
        valid_dataset = ReadabilityDataset(valid, tokenizer)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.valid_batch_size,
            pin_memory=True,
            drop_last=False,
            num_workers=config.num_workers
        )
        model.load_state_dict(torch.load("{config.save_path}_{fold}", map_location=config.device))

        valid_out, valid_tar, valid_ids = oof_out(valid_loader, model, config.device)

        oof_id.append(valid_ids)
        oof_tar.append(valid_tar)
        oof_pred.append(valid_out)
        oof_fold.append([fold]*len(valid_ids))

    return oof_pred, oof_tar, oof_id, oof_fold


def save_log(loss_cv, rmse, config):
    with open(f'/logs.txt', 'w') as f:
        f.write(f'##### VALID SCORES ##############\n')
        f.write(f'RMSE Mean : {np.mean(loss_cv)} \n')
        f.write(f'RMSE Fold Wise : {loss_cv}\n')
        f.write(f'OOF RMSE : {rmse} \n\n')
        f.write(f"######## MODEL PARAMS ##########\n")
        f.write(f'NUM EPOCHS : {config.epochs} \n')
        f.write(f'LR : {config.learning_rate}\n')
        f.write(f"Batch_size : {config.train_batch_size}\n")
        f.write(f"Weight Decay : {config.weight_decay}\n")
        f.write(f"Scheduler : {config.scheduler} \n")
        f.write(f"Max Length : {config.max_length}\n")
        f.write(f"Differential_LR :{config.use_diff_lr}\n")
        f.write(f"Optimizer : AdamW\n")
        f.write(f"Ltype : {config.loss_type}\n")
