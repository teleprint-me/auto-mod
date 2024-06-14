import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class Config:
    n_ctx: int = 1024
    n_embed: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    vocab_size: int = 50257

    def __init__(self, path: Optional[Union[str, Path]]):
        self.path = Path(path) if isinstance(path, str) else path
        assert (path / "config.json").exists(), "Expected dir path to config.json"

    def load(self) -> None:
        with open(self.path, mode="r") as file:
            data = json.load(file)
            self.n_ctx = data.get("n_ctx", 1024)
            self.n_embed = data.get("n_embed", 768)
            self.n_head = data.get("n_head", 12)
            self.n_layer = data.get("n_layer", 12)
            self.n_positions = data.get("n_positions", 1024)
            self.vocab_size = data.get("vocab_size", 50257)

    def save(self) -> None:
        with open(self.path, mode="w") as file:
            data = dict(
                n_ctx=self.n_ctx,
                n_embed=self.n_embed,
                n_head=self.n_head,
                n_layer=self.n_layer,
                n_positions=self.n_positions,
                vocab_size=self.vocab_size,
            )
            json.dump(data, file)
