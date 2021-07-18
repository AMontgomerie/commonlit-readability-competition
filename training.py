import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import List, Tuple

from utils import AverageMeter


DEFAULT_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1., 1)]


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
        fold: int
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

                # Reducing eval steps
                for rmse, period in self.eval_schedule:
                    if self.valid_score >= rmse:
                        self.eval_steps = period
                        break

                if self.valid_score < best_score:
                    best_score = self.valid_score
                    torch.save(
                        self.model.state_dict(),
                        self.save_path
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
            input_ids = batch[0]
            attention_mask = batch[1]
            targets = batch[2]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            output = self.model(input_ids, attention_mask)
            loss = criterion(output, targets.view(-1, 1))
            loss_score.update(loss.detach().item(), batch_size)

        return np.sqrt(loss_score.avg)
