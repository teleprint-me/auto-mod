import os

from argparse import ArgumentParser, Namespace
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace


def get_arguments() -> Namespace:
    parser = ArgumentParser(description="Train a tokenizer model")
    parser.add_argument(
        "-d",
        "--dir-path",
        default="data/wikitext-103-raw-v1",
        help="The directory to the dataset. Default is 'data/wikitext-103-raw-v1'.",
    )
    parser.add_argument(
        "-t",
        "--train-model",
        action="store_true",
        help="Train the tokenizer model. Default is False.",
    )

    return parser.parse_args()


def get_raw_file_paths(path: Path) -> list[str]:
    file_paths = []
    for entry in os.scandir(str(path)):
        if Path(entry.path).suffix == ".raw":
            file_paths.append(entry.path)
    return file_paths


def train_tokenizer_model(data_path: Path, tokenizer_path: Path) -> Tokenizer:
    raw_files = get_raw_file_paths(data_path)

    # UNK (Unknown), CLS (Classification), PAD (Padding), MASK ()
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(raw_files, trainer)
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")
    return tokenizer


def main():
    args = get_arguments()
    path = Path(args.dir_path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} does not exist!")

    tokenizer_path = path / "tokenizer.json"
    if args.train_model:
        tokenizer = train_tokenizer_model(path, tokenizer_path)
    elif tokenizer_path.is_file():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        raise FileNotFoundError(f"Oops! Does '{tokenizer_path}' exist?")

    input_text = "Hello, y'all! How are you 😁 ?"
    output = tokenizer.encode(input_text)
    print("text:", input_text)
    print("tokens:", output.tokens)
    print("ids:", output.ids)
    print("offsets:", output.offsets)


if __name__ == "__main__":
    main()
