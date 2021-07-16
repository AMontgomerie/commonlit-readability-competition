import argparse


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
        "--extra_attention_head",
        dest="extra_attention_head",
        action="store_true"
    ),
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
