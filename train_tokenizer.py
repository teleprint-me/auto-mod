import os
import pyarrow.parquet as pq

from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

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

# # UNK -> Unknown
# # CLS -> Classification
# # PAD -> Padding
# # MASK ->
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)
tokenizer.train(raw_files, trainer)
tokenizer.save(f"{path}/tokenizer.json")
