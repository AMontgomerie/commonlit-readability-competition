import argparse
import os
import pandas as pd
import tez
import transformers
import numpy as np
import random
import torch
import itertools
from sklearn import metrics
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler


class AttentionHead(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        output = self.dropout(context_vector)
        return output


class CommonlitModel(tez.Model):
    def __init__(self, model_name, num_train_steps, steps_per_epoch, learning_rate, loss_type):
        super().__init__()
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"
        self.loss_type = loss_type
        hidden_dropout_prob: float = 0.0
        layer_norm_eps: float = 1e-7
        config = transformers.AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
            }
        )
        self.transformer = transformers.AutoModel.from_pretrained(model_name, config=config)
        self.attention = AttentionHead(in_size=config.hidden_size, hidden_size=config.hidden_size)
        self.regressor = nn.Linear(config.hidden_size * 2, 2)

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )
        return sch

    def loss(self, outputs, targets, standard_errors):
        if self.loss_type == "rmse":
            return torch.sqrt(nn.MSELoss()(outputs, targets))
        elif self.loss_type == "custom":
            return torch.sqrt((torch.square(outputs - targets) * standard_errors).sum())
        else:
            raise ValueError(f"unrecognized loss function: {self.loss_type}")

    def monitor_metrics(self, outputs, targets):
        outputs = outputs.cpu().detach().numpy()[:, 1].ravel()
        targets = targets.cpu().detach().numpy()[:, 1].ravel()
        mse = metrics.mean_squared_error(targets, outputs)
        rmse = np.sqrt(mse)
        return {"rmse": rmse, "mse": mse}

    def forward(self, ids1, mask1, ids2, mask2, targets=None, standard_errors=None):
        output1 = self.transformer(ids1, mask1)
        output2 = self.transformer(ids2, mask2)
        output1 = self.attention(output1.last_hidden_state)
        output2 = self.attention(output2.last_hidden_state)
        output = torch.cat((output1, output2), dim=1)
        output = self.regressor(output)
        loss = self.loss(output, targets, standard_errors)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc


class CommonlitDataset:
    def __init__(self, excerpts, standard_error_dict, target_dict, tokenizer, max_len,
                 num_samples=None):
        self.excerpts = excerpts
        self.target_dict = target_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_samples = num_samples
        self.standard_error_dict = standard_error_dict

    def __len__(self):
        return len(self.excerpts)

    def __getitem__(self, item):
        text1 = str(self.excerpts[item][1])
        text2 = str(self.excerpts[item][0])
        target = [self.target_dict[text2], self.target_dict[text1]]
        standard_errors = [self.standard_error_dict[text2], self.standard_error_dict[text1]]
        inputs1 = self.tokenizer(text1, max_length=self.max_len,
                                 padding="max_length", truncation=True)
        inputs2 = self.tokenizer(text2, max_length=self.max_len,
                                 padding="max_length", truncation=True)
        ids1 = inputs1["input_ids"]
        mask1 = inputs1["attention_mask"]
        ids2 = inputs2["input_ids"]
        mask2 = inputs2["attention_mask"]
        return {
            "ids1": torch.tensor(ids1, dtype=torch.long),
            "mask1": torch.tensor(mask1, dtype=torch.long),
            "ids2": torch.tensor(ids2, dtype=torch.long),
            "mask2": torch.tensor(mask2, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.float),
            "standard_errors": torch.tensor(standard_errors, dtype=torch.float)
        }


class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--learning_rate", type=float, default=3e-5, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--max_len", type=int, default=256, required=False)
    parser.add_argument("--output_folder", type=str, default="pairwise_model/")
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--fulldata", action="store_true")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--early_stopping_patience", type=int, default=5, required=False)
    parser.add_argument("--loss_type", type=str, default="rmse", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.fulldata:
        seed_everything(42)
    os.makedirs(args.output_folder, exist_ok=True)
    output_path = os.path.join(
        args.output_folder,
        f"{args.model.replace('/',':')}__fold_{args.fold}.bin",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    df = pd.read_csv("/content/drive/MyDrive/commonlit/train_folds.csv")
    # base string is excerpt where target is 0 in the dataframe
    base_string = df.loc[df.target == 0, "excerpt"].values[0]
    # create dictionary out of excerpt and target columns from dataframe
    target_dict = dict(zip(df.excerpt.values.tolist(), df.target.values.tolist()))
    standard_error_dict = dict(zip(df.excerpt.values.tolist(), df.standard_error.tolist()))
    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)
    training_pairs = list(itertools.combinations(df_train.excerpt.values.tolist(), 2))
    #training_pairs = [(base_string, k) for k in df_train.excerpt.values.tolist()] + training_pairs
    # randomize training_pairs
    random.shuffle(training_pairs)
    validation_pairs = [(base_string, k) for k in df_valid.excerpt.values.tolist()]
    train_dataset = CommonlitDataset(
        excerpts=training_pairs,
        target_dict=target_dict,
        standard_error_dict=standard_error_dict,
        tokenizer=tokenizer,
        max_len=args.max_len,
        num_samples=args.num_samples,
    )
    valid_dataset = CommonlitDataset(
        excerpts=validation_pairs,
        target_dict=target_dict,
        standard_error_dict=standard_error_dict,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    n_train_steps = int(args.num_samples / args.batch_size * args.epochs)
    model = CommonlitModel(
        model_name=args.model,
        num_train_steps=n_train_steps,
        learning_rate=args.learning_rate,
        steps_per_epoch=args.num_samples / args.batch_size,
        loss_type=args.loss_type

    )
    es = tez.callbacks.EarlyStopping(
        monitor="valid_rmse",
        model_path=output_path,
        save_weights_only=True,
        patience=args.early_stopping_patience
    )
    train_sampler = RandomSampler(
        train_dataset,
        num_samples=args.num_samples
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_sampler=train_sampler,
        train_shuffle=False,
        valid_shuffle=False,
        train_bs=args.batch_size,
        valid_bs=64,
        device="cuda",
        epochs=args.epochs,
        callbacks=[es],
        fp16=True,
    )
