import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

from train_model import CommonLitDataset, train

LEARNING_RATE = 1e-5
EPOCHS = 8
BATCH_SIZE = 16
WARMUP_STEPS = 100
CHECKPOINT = "bert-base-cased"
RANDOM_SEED = 0
WEIGHT_DECAY = 0.4
DROPOUT = 0.4

if __name__ == "__main__":
    data = pd.read_csv("../data/train.csv")
    tokenizer = BertTokenizerFast.from_pretrained(CHECKPOINT)
    train_set, test_set = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED
    )
    train(
        train_set,
        test_set,
        BertForSequenceClassification,
        tokenizer,
        scheduler_type="constant",
    )