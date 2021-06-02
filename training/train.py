import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)
from typing import Mapping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

CHECKPOINT = "roberta-large"
RANDOM_SEED = 0
DEVICE = torch.device("cuda")
torch.manual_seed(RANDOM_SEED)

DEFAULT_CONFIG = {
    "attention_dropout": 0.2,
    "batch_size": 8,
    "epochs": 10,
    "hidden_dropout": 0.2,
    "learning_rate": 1e-5,
    "save_path": "output",
    "warmup_steps": 150,
    "weight_decay": 0.1,
    "target_sampling": False,
}


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerFast,
        sample_targets=False,
    ) -> None:
        self.texts = data.excerpt.to_list()
        self.targets = data.target.to_list()
        self.tokenizer = tokenizer
        self.standard_errors = data.standard_error.to_list()
        self.sample_targets = sample_targets

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Mapping[str, torch.tensor]:
        text = self.texts[index]
        encoded_inputs = self.tokenizer(
            text,
            max_length=330,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )

        encoded_inputs["input_ids"] = torch.squeeze(encoded_inputs["input_ids"]).to(
            DEVICE
        )
        encoded_inputs["attention_mask"] = torch.squeeze(
            encoded_inputs["attention_mask"]
        ).to(DEVICE)

        target = self.targets[index]

        if self.sample_targets:
            std_error = self.standard_errors[index]
            target = self.add_target_sampling(target, std_error)

        encoded_inputs["labels"] = torch.tensor(target).to(DEVICE)
        return encoded_inputs

    def add_target_sampling(self, target, std):
        return np.random.normal(target, std)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.2,
        help="the dropout rate in self attention layers",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="the number of examples per minibatch"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="the number of epochs to run for"
    )
    parser.add_argument(
        "--hidden_dropout",
        type=float,
        default=0.2,
        help="the dropout rate in hidden layers",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="the max learning rate"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output",
        help="where to save the trained models",
    )
    parser.add_argument(
        "--target_sampling", dest="target_sampling", action="store_true"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=150,
        help="the number of warmup steps at the beginning of training",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="the rate of weight decay in AdamW",
    )
    parser.set_defaults(target_sampling=False)
    return parser.parse_args()


def build_config(params):
    return {
        "attention_dropout": params.attention_dropout,
        "batch_size": params.batch_size,
        "epochs": params.epochs,
        "hidden_dropout": params.hidden_dropout,
        "learning_rate": params.learning_rate,
        "save_path": params.save_path,
        "target_sampling": params.target_sampling,
        "warmup_steps": params.warmup_steps,
        "weight_decay": params.weight_decay,
    }


def train(
    fold: int, train_set: Dataset, valid_set: Dataset, config: Mapping
) -> PreTrainedModel:
    model = RobertaForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=1,
        hidden_dropout_prob=config["hidden_dropout"],
        attention_probs_dropout_prob=config["attention_dropout"],
    ).to(DEVICE)
    model.train()
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"])
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=len(train_loader) * config["epochs"],
    )

    for epoch in range(1, config["epochs"] + 1):

        total_rmse = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(**batch)
            loss = torch.sqrt(criterion(torch.squeeze(output.logits), batch["labels"]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_rmse += compute_rmse(batch["labels"], output.logits.detach())

        current_steps = (epoch - 1) * len(train_loader) + step
        train_rmse = total_rmse / len(train_loader)
        valid_rmse = evaluate(model, valid_set, config["batch_size"])
        print(
            f"Fold: {fold} | Step: {current_steps} Epoch: {epoch} | Train RMSE: {train_rmse} | Valid RMSE: {valid_rmse}"
        )

    return model


@torch.no_grad()
def evaluate(model: PreTrainedModel, data: Dataset, batch_size) -> float:
    model.eval()
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)
    total_rmse = 0

    for batch in data_loader:
        output = model(**batch)
        total_rmse += compute_rmse(batch["labels"], output.logits.detach())

    return total_rmse / len(data_loader)


def compute_rmse(targets: torch.tensor, preds: torch.tensor) -> float:
    rmse = mean_squared_error(targets.cpu(), preds.cpu(), squared=False)
    return rmse.item()


def save(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    fold: int,
    output_dir: str = "output",
):
    path = os.path.join(output_dir, f"model_{fold}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_cv(config: Mapping = DEFAULT_CONFIG) -> float:
    tokenizer = RobertaTokenizerFast.from_pretrained(CHECKPOINT)
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train_folds.csv")
    data = pd.read_csv(path)

    if "folds" in config:
        folds = config["folds"]
    else:
        folds = len(data.kfold.unique())

    scores = []

    print(f"Training {CHECKPOINT} for {folds} folds with:")
    print(config)

    for fold in range(folds):
        train_set = CommonLitDataset(
            data[data.kfold != fold],
            tokenizer,
            sample_targets=config["target_sampling"],
        )
        valid_set = CommonLitDataset(data[data.kfold == fold], tokenizer)
        trained_model = train(fold, train_set, valid_set, config)
        rmse = evaluate(trained_model, valid_set, config["batch_size"])
        save(trained_model, tokenizer, fold, output_dir=config["save_path"])
        scores.append(rmse)
        torch.cuda.empty_cache()
        del trained_model
        gc.collect()

    return sum(scores) / len(scores)


if __name__ == "__main__":
    params = parse_args()
    config = build_config(params)
    cv_score = train_cv(config)
    print(f"CV score: {cv_score}")
