#!/usr/bin/env python
"""
data2txt.py - convert parquet files to plaintext
"""

import argparse
import os
import multiprocessing

from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm


def parquet_to_plaintext(path: Path) -> None:
    # Convert parquet files into plaintext files
    dataset = load_dataset(str(path))  # DatasetDict
    for index, subset in enumerate(dataset.items()):  # int, tuple
        name, rows = subset  # str, Dataset
        print(f"Converting subset {name} to plaintext")

        # Concatenate all text features for a given subset
        text = ""
        for row in tqdm(rows, total=len(rows)):  # str
            text += "\n".join(f"{row['text'].strip().split('\n')}")

        # Write the concatenated text to a plaintext file
        output_file_path = path / f"{name}-{index:05}-of-{len(dataset.keys()):05}.txt"
        print(f"Writing {output_file_path}")

        try:
            with open(output_file_path, "w") as f:
                f.write(text)
        except (OSError, IOError) as e:
            print(f"Error while writing to file '{output_file_path}': {e}")


def get_raw_file_paths(path: Path) -> list[str]:
    return [
        ent.path for ent in os.scandir(str(path)) if Path(ent.path).suffix == ".txt"
    ]


def main():
    parser = argparse.ArgumentParser(help="Convert parquet files to plaintext")
    parser.add_argument(
        "-d",
        "--dir-path",
        default="data/wikitext-103-raw-v1",
        help="The directory to the dataset",
    )
    args = parser.parse_args()

    path = Path(args.dir_path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} does not exist!")

    # Load the dataset
    parquet_to_plaintext(path)
    raw_files = get_raw_file_paths(path)
    print(raw_files)


if __name__ == "__main__":
    main()
