import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


class ConfigProperties:
    properties = {
        "n_ctx": 1024,
        "n_embed": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "vocab_size": 50257,
    }


@dataclass
class Config:
    def __init__(self, path: Union[str, Path]):
        # Set config path
        self._path = Path(path) if isinstance(path, str) else path

        # Deep copy default model parameters?
        self._data = ConfigProperties.properties.copy()

        # Initialize default key-value pairs
        for key, value in self._data.items():
            setattr(self, key, value)

    @property
    def path(self) -> Path:
        return self._path / "config.json"

    @property
    def data(self) -> dict[str, object]:
        return self._data

    def load(self) -> None:
        with open(self.path, mode="r") as file:
            # Override internal data
            self._data = json.load(file)

            for key, value in self._data.items():
                setattr(self, key, value)

    def save(self, **kwargs) -> None:
        with open(self.path, mode="w") as file:
            if kwargs:
                self._data.update(kwargs)

            json.dump(self._data, file)
