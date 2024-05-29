import os

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, StripAccents

path = Path("data/wikitext-103-raw-v1")
if not path.is_dir():
    raise FileNotFoundError(f"{path} does not exist!")


def get_raw_file_paths(path: Path) -> list[str]:
    file_paths = []
    for entry in os.scandir(str(path)):
        if Path(entry.path).suffix == ".raw":
            file_paths.append(entry.path)
    return file_paths


raw_files = get_raw_file_paths(path)

# UNK -> Unknown
# CLS -> Classification
# PAD -> Padding
# MASK ->
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([NFD(), StripAccents()])
tokenizer.train(raw_files, trainer)
tokenizer.save(path / "tokenizer.json")
