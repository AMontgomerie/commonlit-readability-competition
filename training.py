import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import List, Tuple
from dataclasses import dataclass

from transformers.models.auto.tokenization_auto import AutoTokenizer

from utils import AverageMeter, seed_everything
from data_prep import ReadabilityDataset
from models import TransformerWithAttentionHead


DEFAULT_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]


@dataclass()
class Config:
    eval_schedule: List[Tuple[float, int]]
    device: torch.device = torch.device('cuda')
    max_length: int = 248
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
    save_path: str = "output"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
        criterion: nn.Module,
        seed: int,
        save_path: str,
        print_step: int = 10,
        eval_schedule: List[Tuple[float, int]] = DEFAULT_SCHEDULE
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.evaluator = Evaluator(self.model)
        self.eval_steps = eval_schedule[0][1]
        self.eval_schedule = eval_schedule
        self.last_eval_step = 0
        self.step = 0
        self.seed = seed
        self.valid_score = 0
        self.print_step = print_step
        self.save_path = save_path

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        best_score: float,
        device: torch.device,
        epoch: int,
        fold: int,
    ) -> float:
        loss_score = AverageMeter()
        self.model.train()

        for bi, batch in enumerate(train_loader):
            batch_size = batch[0].shape[0]
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            targets = batch[2].to(device)

            output = self.model(input_ids, attention_mask)
            loss = self.criterion(output, targets.view(-1, 1))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_score.update(loss.detach().item(), batch_size)

            if self.scheduler is not None:
                self.scheduler.step()

            if bi % self.print_step == 0:
                print(f"Step {bi} Train Loss : {loss_score.avg}")

            if self.step >= self.last_eval_step + self.eval_steps:
                self.last_eval_step = self.step
                self.valid_score = self.evaluator.evaluate(valid_loader, device, self.criterion)
                print(
                    f"Valid Score at Step {bi} for Epoch {epoch+1}: {self.valid_score}, Step Size: {self.eval_steps}"
                )

                for rmse, period in self.eval_schedule:
                    if self.valid_score >= rmse:
                        self.eval_steps = period
                        break

                if self.valid_score < best_score:
                    best_score = self.valid_score
                    torch.save(
                        self.model.state_dict(),
                        f"{self.save_path}_{fold}"
                    )
                    print(f'Best model found for epoch {epoch+1} for step {bi}')

            self.step += 1

        return best_score


@torch.no_grad()
class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(
        self,
        dataloader: DataLoader,
        device: torch.device,
        criterion: nn.Module
    ) -> float:
        self.model.eval()
        loss_score = AverageMeter()

        for batch in dataloader:
            batch_size = batch[0].shape[0]
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            targets = batch[2].to(device)
            output = self.model(input_ids, attention_mask)
            loss = criterion(output, targets.view(-1, 1))
            loss_score.update(loss.detach().item(), batch_size)

        return np.sqrt(loss_score.avg)


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
