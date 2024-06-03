from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import frontmatter

from .constants import GGUFMetadataKeys
from .huggingface_hub import HFHubModel


@dataclass
class GGUFMetadata:
    # Authorship GGUFMetadata to be written to GGUF KV Store
    name: Optional[str] = None
    basename: Optional[str] = None
    finetune: Optional[str] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    version: Optional[str] = None
    base_version: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    license_name: Optional[str] = None
    license_link: Optional[str] = None
    source_url: Optional[str] = None
    source_repo: Optional[str] = None
    parameter_size_class: Optional[str] = None
    tags: Optional[list[str]] = None
    language: Optional[list[str]] = None
    datasets: Optional[list[str]] = None

    @staticmethod
    def load(
        metadata_override_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        model_name: Optional[str] = None,
    ) -> GGUFMetadata:
        # This grabs as many contextual authorship metadata as possible from the model repository
        # making any conversion as required to match the gguf kv store metadata format
        # as well as giving users the ability to override any authorship metadata that may be incorrect

        # Create a new GGUFMetadata instance
        metadata = GGUFMetadata()

        # load model folder model card if available
        # Reference Model Card GGUFMetadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
        model_card = GGUFMetadata.load_model_card(model_path)

        for key, value in model_card.items():
            if key == "model_name" and isinstance(value, list) and len(value) >= 1:
                setattr(metadata, "name", value[0].get("name"))
            elif key == "model_name":
                setattr(metadata, "name", value)
            elif key == "model_creator":
                setattr(metadata, "author", value)

            # For attributes like tags, datasets and language that can be lists
            elif key in ["tags", "datasets", "language"]:
                setattr(metadata, key, [])

            # If the attribute is not found in model card or it's empty string (""),
            # we use metadata[key] to preserve its existing value.
            elif not value:
                setattr(metadata, key, getattr(metadata, key))

            else:
                setattr(metadata, key, value)

        # load huggingface parameters if available
        hf_params = GGUFMetadata.load_huggingface_parameters(model_path)
        hf_name_or_path = hf_params.get("_name_or_path")
        if metadata.name is None and hf_name_or_path is not None:
            metadata.name = Path(hf_name_or_path).name
        if metadata.source_repo is None and hf_name_or_path is not None:
            metadata.source_repo = Path(hf_name_or_path).name

        # Use Directory Folder Name As Fallback Name
        if metadata.name is None:
            if model_path is not None and model_path.exists():
                metadata.name = model_path.name

        # GGUFMetadata Override File Provided
        if metadata_override_path is not None:
            override = GGUFMetadata.load_metadata_override(metadata_override_path)

            for key, value in vars(GGUFMetadataKeys.General).items():
                setattr(metadata, key, override.get(key, getattr(metadata, key)))

        # Direct GGUFMetadata Override (via direct cli argument)
        if model_name is not None:
            metadata.name = model_name

        return metadata

    @staticmethod
    def load_metadata_override(metadata_override_path: Optional[Path]):
        if metadata_override_path is None or not metadata_override_path.exists():
            return {}

        with open(metadata_override_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_model_card(model_path: Optional[Path]):
        if model_path is None or not model_path.exists():
            return {}

        model_card_path = model_path / "README.md"

        if not model_card_path.exists():
            return {}

        with open(model_card_path, "r", encoding="utf-8") as f:
            return frontmatter.load(f)

    @staticmethod
    def load_huggingface_parameters(model_path: Optional[Path]):
        if model_path is None or not model_path.exists():
            return {}

        config_path = model_path / "config.json"

        if not config_path.exists():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_remote_model_card(hub_model: HFHubModel):
        pass
