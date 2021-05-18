import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM

MAX_LEN = 330
BATCH_SIZE = 32
EN_FR = "Helsinki-NLP/opus-mt-en-fr"
FR_EN = "Helsinki-NLP/opus-mt-fr-en"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

forward_tokenizer = MarianTokenizer.from_pretrained(EN_FR)
forward_model = AutoModelForSeq2SeqLM.from_pretrained(EN_FR).to(DEVICE)
backward_tokenizer = MarianTokenizer.from_pretrained(FR_EN)
backward_model = AutoModelForSeq2SeqLM.from_pretrained(FR_EN).to(DEVICE)


class TranslationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def back_translate_all(texts):
    back_translations = []
    dataset = TranslationDataset(texts)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
    progress = 0

    for text_batch in dataloader:
        translated = back_translate(text_batch)
        back_translations += translated
        progress += len(text_batch)

        if progress % (len(dataloader) / 10) == 0:
            print(f"progress: {progress}/{len(texts)}")

    return back_translations


def back_translate(texts):
    english_inputs = forward_tokenizer(
        texts, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
    ).to(DEVICE)
    french_outputs = forward_model.generate(english_inputs["input_ids"])
    decoded_french = forward_tokenizer.batch_decode(
        french_outputs, skip_special_tokens=True
    )
    french_inputs = backward_tokenizer(
        decoded_french, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
    ).to(DEVICE)
    english_outputs = backward_model.generate(french_inputs["input_ids"])
    decoded_english = backward_tokenizer.batch_decode(
        english_outputs, skip_special_tokens=True
    )
    return decoded_english


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    back_translations = back_translate_all(train.excerpt.to_list())
    back_translations_df = pd.DataFrame({"translated_text": back_translations})
    back_translations_df.to_csv("fr_back_translations.csv", index=False)