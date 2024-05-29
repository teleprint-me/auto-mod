#!/usr/bin/env python
"""
data2txt.py - convert parquet files to plaintext
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm


def parquet_to_plaintext(path: Path) -> None:
    dataset: DatasetDict = load_dataset(str(path))

    def process_subset(data: tuple[str, Dataset]):
        name, subset = data
        text = ""

        for row in tqdm(subset, total=len(subset)):
            text += row["text"]

        output_file_path = path / f"wiki-{name}.raw"

        print(f"Writing {output_file_path}")

        try:
            with open(output_file_path, "w") as f:
                f.write(text)

        except (OSError, IOError) as e:
            print(f"Error while writing to file '{output_file_path}': {e}")

    with ThreadPoolExecutor(max_workers=cpu_count()) as pool:
        pool.map(process_subset, dataset.items())


def get_raw_file_paths(path: Path) -> list[str]:
    file_paths = []
    for entry in os.scandir(str(path)):
        if Path(entry.path).suffix == ".raw":
            file_paths.append(entry.path)
    return file_paths


def main():
    parser = argparse.ArgumentParser()

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

    for raw in get_raw_file_paths(path):
        print(raw)


if __name__ == "__main__":
    main()
