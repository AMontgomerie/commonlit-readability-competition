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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from typing import Mapping
from sklearn.metrics import mean_squared_error

DEVICE = torch.device("cuda")

DEFAULT_CONFIG = {
    "attention_dropout": 0.2,
    "batch_size": 8,
    "checkpoint": "roberta-large",
    "early_stopping": True,
    "epochs": 10,
    "eval_steps": 50,
    "eval_style": "epochs",
    "hidden_dropout": 0.2,
    "learning_rate": 1e-5,
    "max_length": 330,
    "random_seed": 0,
    "save_path": "output",
    "scheduler": "linear",
    "target_sampling": False,
    "warmup_steps": 150,
    "weight_decay": 0.1,
}


class CommonLitDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int,
        sample_targets: bool = False,
    ) -> None:
        self.texts = data.excerpt.to_list()
        self.targets = data.target.to_list()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.standard_errors = data.standard_error.to_list()
        self.sample_targets = sample_targets

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Mapping[str, torch.tensor]:
        text = self.texts[index]
        encoded_inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
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
        "--checkpoint",
        type=str,
        default="roberta-large",
        help="which pretrained model and tokenizer to use",
    )
    parser.add_argument(
        "--early_stopping", dest="early_stopping", action="store_true"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="the number of epochs to run for"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="how often to evaluate and save the model",
    )
    parser.add_argument(
        "--eval_style",
        type=str,
        default="epochs",
        help="whether to evaluate every epoch or per x number of steps",
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
        "--max_length",
        type=int,
        default=330,
        help="the maximum sequence length in tokens",
    ),
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="the random seed",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output",
        help="where to save the trained models",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        help="choose a scheduler from [constant, linear, cosine]",
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
        "checkpoint": params.checkpoint,
        "early_stopping": params.early_stopping,
        "epochs": params.epochs,
        "eval_steps": params.eval_steps,
        "eval_style": params.eval_style,
        "hidden_dropout": params.hidden_dropout,
        "learning_rate": params.learning_rate,
        "max_length": params.max_length,
        "save_path": params.save_path,
        "scheduler": params.scheduler,
        "target_sampling": params.target_sampling,
        "warmup_steps": params.warmup_steps,
        "weight_decay": params.weight_decay,
    }


def get_model(checkpoint, hidden_dropout, attention_dropout):
    if checkpoint.startswith("xlnet") or checkpoint.startswith("transfo-xl"):
        print("Assigning dropout=hidden_dropout.")
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=1,
            dropout=hidden_dropout
        ).to(DEVICE)

    elif checkpoint.startswith("funnel-transformer"):
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=1,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
        ).to(DEVICE)

    else:
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=1,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout,
        ).to(DEVICE)


def train(
    fold: int, train_set: Dataset, valid_set: Dataset, config: Mapping
) -> AutoModelForSequenceClassification:
    model = get_model(config["checkpoint"], config["hidden_dropout"], config["attention_dropout"])
    model.train()
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"])
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_scheduler(
        config["scheduler"],
        optimizer,
        warmup_steps=config["warmup_steps"],
        total_steps=len(train_loader) * config["epochs"],
    )

    best_rmse = 1.0
    total_rmse = 0

    for epoch in range(1, config["epochs"] + 1):

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch["input_ids"], batch["attention_mask"])
            loss = torch.sqrt(criterion(torch.squeeze(output.logits), batch["labels"]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_rmse += compute_rmse(batch["labels"], output.logits.detach())
            current_steps = (epoch - 1) * len(train_loader) + step

            saved = False
            if (
                config["eval_style"] == "steps"
                and current_steps != 0
                and current_steps % config["eval_steps"] == 0
            ):
                train_rmse = total_rmse / config["eval_steps"]
                valid_rmse = evaluate(model, valid_set, config["batch_size"])
                total_rmse = 0

                if config["early_stopping"] and valid_rmse < best_rmse:
                    save(
                        model, train_set.tokenizer, fold, output_dir=config["save_path"]
                    )
                    best_rmse = valid_rmse
                    saved = True

                print(
                    f"Fold: {fold} | "
                    f"Step: {current_steps} | "
                    f"Epoch: {epoch} | "
                    f"Train RMSE: {train_rmse} | "
                    f"Valid RMSE: {valid_rmse} | "
                    f"{'Model saved' if saved else ''}"
                )

        if config["eval_style"] == "epochs":
            train_rmse = total_rmse / len(train_loader)
            valid_rmse = evaluate(model, valid_set, config["batch_size"])
            total_rmse = 0

            if config["early_stopping"] and valid_rmse < best_rmse:
                save(model, train_set.tokenizer, fold, output_dir=config["save_path"])
                best_rmse = valid_rmse
                saved = True

            print(
                f"Fold: {fold} | "
                f"Step: {current_steps} | "
                f"Epoch: {epoch} | "
                f"Train RMSE: {train_rmse} | "
                f"Valid RMSE: {valid_rmse} | "
                f"{'Model saved' if saved else ''}"
            )

    if config["eval_style"] == "steps" or not config["early_stopping"]:
        valid_rmse = evaluate(model, valid_set, config["batch_size"])
        saved = False
        if valid_rmse < best_rmse:
            save(model, train_set.tokenizer, fold, output_dir=config["save_path"])
            best_rmse = valid_rmse
            saved = True

    print(
        f"Fold {fold} | Training Complete | Valid RMSE: {valid_rmse} | {'Model saved' if saved else ''}"
    )

    return model, best_rmse


def get_scheduler(scheduler_type, optimizer, warmup_steps, total_steps):
    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        raise ValueError(f"{scheduler_type} is not an available scheduler type.")


@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification, data: Dataset, batch_size
) -> float:
    model.eval()
    data_loader = DataLoader(data, shuffle=False, batch_size=batch_size)
    total_rmse = 0

    for batch in data_loader:
        output = model(batch["input_ids"], batch["attention_mask"])
        total_rmse += compute_rmse(batch["labels"], output.logits.detach())

    return total_rmse / len(data_loader)


def compute_rmse(targets: torch.tensor, preds: torch.tensor) -> float:
    rmse = mean_squared_error(targets.cpu(), preds.cpu(), squared=False)
    return rmse.item()


def save(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    fold: int,
    output_dir: str = "output",
):
    path = os.path.join(output_dir, f"model_{fold}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train_cv(config: Mapping = DEFAULT_CONFIG) -> float:
    seed_everything(config["random_seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train_folds.csv")
    data = pd.read_csv(path)

    if "folds" in config:
        folds = config["folds"]
    else:
        folds = data.kfold.unique()

    scores = []

    print(f"Training {config['checkpoint']} for {len(folds)} folds in reverse order with:")
    print(config)

    for fold in reversed(folds):
        train_set = CommonLitDataset(
            data[data.kfold != fold],
            tokenizer,
            config["max_length"],
            sample_targets=config["target_sampling"],
        )
        valid_set = CommonLitDataset(
            data[data.kfold == fold],
            tokenizer,
            config["max_length"],
        )
        trained_model, rmse = train(fold, train_set, valid_set, config)
        scores.append(rmse)
        torch.cuda.empty_cache()
        del trained_model
        gc.collect()

    return sum(scores) / len(scores)


def seed_everything(seed=0):
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    params = parse_args()
    config = build_config(params)
    cv_score = train_cv(config)
    print(f"CV score: {cv_score}")
