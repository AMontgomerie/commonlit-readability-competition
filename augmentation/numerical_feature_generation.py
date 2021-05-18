import en_core_web_sm
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import string
from textstat import textstat

nlp = en_core_web_sm.load()
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
PUNCTUATION = list(string.punctuation)
POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
]
RANDOM_SEED = 0


def generate_features(data):
    feature_data = []

    for text in data:
        features = preprocess_text(text)
        feature_data.append(features)

    return pd.DataFrame(feature_data)


def preprocess_text(text):
    simplified_text = simplify_punctuation(text)

    features = {
        "flesch_reading_ease": textstat.flesch_reading_ease(simplified_text),
        "smog_index": textstat.smog_index(simplified_text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(simplified_text),
        "coleman_liau_index": textstat.coleman_liau_index(simplified_text),
        "automated_readability_index": textstat.automated_readability_index(
            simplified_text
        ),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(
            simplified_text
        ),
        "difficult_words": textstat.difficult_words(simplified_text),
        "linsear_write_formula": textstat.linsear_write_formula(simplified_text),
        "gunning_fog": textstat.gunning_fog(simplified_text),
        "text_standard": textstat.text_standard(simplified_text, float_output=True),
        "mean_parse_tree_depth": get_mean_parse_tree_depth(text),
        "mean_ents_per_sentence": get_mean_ents_per_sentence(text),
        "total_ents": get_total_ents(text),
        "total_chars": get_num_chars(text),
        "total_words": get_num_words(text),
        "chars_per_word": get_mean_chars_per_word(text),
        "total_sentences": get_num_sentences(text),
        "words_per_sentence": get_mean_words_per_sentence(text),
        "nonstop_word_count": get_mean_nonstop_word_count(text),
        "nonstop_char_count": get_mean_nonstop_char_length(text),
        "nonstop_token_proportion": get_nonstop_proportion(text),
    }

    features.update(get_mean_pos_tags(text))

    return features


def simplify_punctuation(text):
    # from https://github.com/shivam5992/textstat/issues/77

    text = re.sub(r"[,:;()\-]", " ", text)  # Override commas, colons, etc to spaces/
    text = re.sub(r"[\.!?]", ".", text)  # Change all terminators like ! and ? to "."
    text = re.sub(r"^\s+", "", text)  # Remove white space
    text = re.sub(r"[ ]*(\n|\r\n|\r)[ ]*", " ", text)  # Remove new lines
    text = re.sub(r"([\.])[\. ]+", ".", text)  # Change all ".." to "."
    text = re.sub(r"[ ]*([\.])", ". ", text)  # Normalize all "."`
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    text = re.sub(r"\s+$", "", text)  # Remove trailing spaces
    return text


def get_mean_parse_tree_depth(text):
    sentences = text.split(".")
    depths = []
    for doc in list(nlp.pipe(sentences)):
        depths += get_parse_tree_depths(doc)
    return np.mean(depths)


def get_parse_tree_depths(doc):
    return [get_depth(token) for token in doc]


def get_depth(token, depth=0):
    depths = [get_depth(child, depth + 1) for child in token.children]
    return max(depths) if len(depths) > 0 else depth


def get_mean_pos_tags(text):
    sentences = text.split(".")
    sentence_counts = make_pos_tag_count_lists(sentences)
    num_sentences = textstat.sentence_count(text)
    mean_pos_tags = calculate_mean_per_tag(sentence_counts, num_sentences)
    return mean_pos_tags


def make_pos_tag_count_lists(sentences):
    sentence_counts = {}
    for doc in list(nlp.pipe(sentences)):
        pos_counts = get_pos_tag_counts(doc)
        for key in pos_counts:
            if key in sentence_counts:
                sentence_counts[key].append(pos_counts[key])
            else:
                sentence_counts[key] = [pos_counts[key]]
    return sentence_counts


def get_pos_tag_counts(doc):
    pos_counts = {}
    pos_tags = [token.pos_ for token in doc]
    for tag in pos_tags:
        if tag in pos_counts:
            pos_counts[tag] += 1
        else:
            pos_counts[tag] = 1
    return pos_counts


def calculate_mean_per_tag(counts, num_sentences):
    mean_pos_tags = {f"mean_{tag.lower()}": 0 for tag in POS_TAGS}
    for key in counts:
        if len(counts[key]) < num_sentences:
            counts[key] += [0] * (num_sentences - len(counts[key]))
        mean_value = round(np.mean(counts[key]), 2)
        mean_pos_tags["mean_" + key.lower()] = mean_value
    return mean_pos_tags


def get_total_ents(text):
    return len(nlp(text).doc.ents)


def get_mean_ents_per_sentence(text):
    return get_total_ents(text) / textstat.sentence_count(text)


def get_mean_chars_per_word(text):
    return get_num_chars(text) / get_num_words(text)


def get_mean_words_per_sentence(text):
    return get_num_words(text) / get_num_sentences(text)


def get_mean_nonstop_char_length(text):
    spans = tokenize_on_stopwords(text)
    return sum([get_num_chars(span) for span in spans]) / len(spans)


def get_mean_nonstop_word_count(text):
    spans = tokenize_on_stopwords(text)
    return sum([get_num_words(span) for span in spans]) / len(spans)


def get_nonstop_proportion(text):
    tokens = nltk.word_tokenize(text)
    nonstop_tokens = [token for token in tokens if token not in STOPWORDS + PUNCTUATION]
    return len(nonstop_tokens) / len(tokens)


def tokenize_on_stopwords(text):
    tokens = nltk.word_tokenize(text)
    spans = []
    current_span = []
    for token in tokens:
        if token not in STOPWORDS + PUNCTUATION:
            current_span.append(token)
        else:
            if len(current_span) > 0:
                spans.append(" ".join(current_span))
            current_span = []
    return spans


def get_num_chars(text):
    return len(text)


def get_num_words(text):
    return len(text.split())


def get_num_sentences(text):
    total = text.count(".") + text.count("?") + text.count("!")
    if total == 0:
        return 1
    else:
        return total


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    numerical_features = generate_features(train.excerpt.to_list())
    numerical_features.to_csv("data/numerical_features.csv", index=False)