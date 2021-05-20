import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import mean_squared_error

LEARNING_RATE = 1e-5
EPOCHS = 8
BATCH_SIZE = 16
WARMUP_STEPS = 100
CHECKPOINT = "bert-base-cased"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RANDOM_SEED = 0
WEIGHT_DECAY = 0.4
DROPOUT = 0.4

tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT)
torch.manual_seed(RANDOM_SEED)


class CommonLitDataset(Dataset):
    def __init__(self, data, tokenizer, sample_targets=False):
        self.texts = data.excerpt.to_list()
        self.targets = data.target.to_list()
        self.standard_errors = data.standard_error.to_list()
        self.tokenizer = tokenizer
        self.sample_targets = sample_targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        encoded_inputs = self.encode_text(text)
        target = self.targets[index]

        if self.sample_targets:
            std_error = self.standard_errors[index]
            target = self.add_target_sampling(target, std_error)

        encoded_inputs["input_ids"] = torch.squeeze(encoded_inputs["input_ids"]).to(
            DEVICE
        )
        encoded_inputs["attention_mask"] = torch.squeeze(
            encoded_inputs["attention_mask"]
        ).to(DEVICE)
        encoded_inputs["labels"] = torch.tensor([target]).to(DEVICE)
        return encoded_inputs

    def encode_text(self, text):
        return self.tokenizer(
            text,
            max_length=330,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )

    def add_target_sampling(self, target, std):
        return np.random.normal(target, std)


def train(train_data, valid_data, model_type, tokenizer, scheduler_type="constant"):
    train = CommonLitDataset(train_data.excerpt, train_data.target, tokenizer)
    valid = CommonLitDataset(valid_data.excerpt, valid_data.target, tokenizer)
    train_loader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
    model = model_type.from_pretrained(
        CHECKPOINT,
        num_labels=1,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    ).to(DEVICE)
    model.train()
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, scheduler_type, len(train_loader) * EPOCHS)
    best_rmse = 9999
    last_save = -1

    for epoch in range(1, EPOCHS + 1):

        total_rmse = 0

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(**batch)
            loss = torch.sqrt(criterion(torch.squeeze(output.logits), batch["labels"]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_rmse += compute_rmse(batch["labels"], output.logits.detach())

        train_rmse = total_rmse / len(train_loader)
        valid_rmse = evaluate(model, valid)
        print(f"Epoch: {epoch} | Train RMSE: {train_rmse} | Valid RMSE: {valid_rmse}")

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            last_save = epoch
            save(model, tokenizer)

    print(f"Last saved at epoch {last_save}")
    return model


def evaluate(model, data):
    model.eval()
    data_loader = DataLoader(data, shuffle=False, batch_size=BATCH_SIZE)
    total_rmse = 0

    with torch.no_grad():
        for batch in data_loader:
            output = model(**batch)
            total_rmse += compute_rmse(batch["labels"], output.logits.detach())

    return total_rmse / len(data_loader)


def compute_rmse(targets, preds):
    rmse = mean_squared_error(targets.cpu(), preds.cpu(), squared=False)
    return rmse.item()


def save(model, tokenizer):
    model.save_pretrained("./output")
    tokenizer.save_pretrained("./output")


def get_scheduler(optimizer, type, total_steps=None):
    if type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS
        )
    elif type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
        )
    elif type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
        )
    else:
        raise ValueError(f"{type} is not a valid scheduler")

    return scheduler