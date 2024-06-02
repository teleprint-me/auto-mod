from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import frontmatter

from .constants import GGUFMetadataKeys


@dataclass
class Metadata:
    # Authorship Metadata to be written to GGUF KV Store
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
    source_hf_repo: Optional[str] = None
    parameter_size_class: Optional[str] = None
    tags: Optional[list[str]] = None
    language: Optional[list[str]] = None
    datasets: Optional[list[str]] = None

    @staticmethod
    def load(
        metadata_override_path: Optional[Path],
        model_path: Optional[Path],
        model_name: Optional[str],
    ) -> Metadata:
        # This grabs as many contextual authorship metadata as possible from the model repository
        # making any conversion as required to match the gguf kv store metadata format
        # as well as giving users the ability to override any authorship metadata that may be incorrect

        # Create a new Metadata instance
        metadata = Metadata()

        # load model folder model card if available
        # Reference Model Card Metadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
        model_card = Metadata.load_model_card(model_path)
        if metadata.name is None:
            if (
                "model-index" in model_card
                and len(model_card["model_name"]) == 1
                and "name" in model_card["model_name"][0]
            ):
                # We check if there is only one model information in the model-index
                # (This is a safe choice in case there is multiple models in one repo in the future)
                metadata.name = model_card["model_name"][0].get("name")
            elif "model_name" in model_card:
                # non huggingface model card standard but notice some model creator using it
                metadata.name = model_card.get("model_name")
        if metadata.license is None:
            metadata.license = model_card.get("license")
        if metadata.license_name is None:
            metadata.license_name = model_card.get("license_name")
        if metadata.license_link is None:
            metadata.license_link = model_card.get("license_link")
        if metadata.author is None:
            # non huggingface model card standard but notice some model creator using it
            metadata.author = model_card.get("model_creator")
        if metadata.tags is None:
            metadata.tags = model_card.get("tags", [])
        if metadata.language is None:
            metadata.language = model_card.get("language", [])
        if metadata.datasets is None:
            metadata.datasets = model_card.get("datasets", [])

        # load huggingface parameters if available
        hf_params = Metadata.load_huggingface_parameters(model_path)
        hf_name_or_path = hf_params.get("_name_or_path")
        if metadata.name is None and hf_name_or_path is not None:
            metadata.name = Path(hf_name_or_path).name
        if metadata.source_hf_repo is None and hf_name_or_path is not None:
            metadata.source_hf_repo = Path(hf_name_or_path).name

        # Use Directory Folder Name As Fallback Name
        if metadata.name is None:
            if model_path is not None and model_path.exists():
                metadata.name = model_path.name

        # Metadata Override File Provided
        # This is based on LLM_KV_NAMES mapping in llama.cpp
        metadata_override = Metadata.load_metadata_override(metadata_override_path)
        metadata.name = metadata_override.get(
            GGUFMetadataKeys.General.NAME, metadata.name
        )  # noqa: E202
        metadata.basename = metadata_override.get(
            GGUFMetadataKeys.General.BASENAME, metadata.basename
        )  # noqa: E202
        metadata.finetune = metadata_override.get(
            GGUFMetadataKeys.General.FINETUNE, metadata.finetune
        )  # noqa: E202
        metadata.author = metadata_override.get(
            GGUFMetadataKeys.General.AUTHOR, metadata.author
        )  # noqa: E202
        metadata.organization = metadata_override.get(
            GGUFMetadataKeys.General.ORGANIZATION, metadata.organization
        )  # noqa: E202
        metadata.version = metadata_override.get(
            GGUFMetadataKeys.General.VERSION, metadata.version
        )  # noqa: E202
        metadata.base_version = metadata_override.get(
            GGUFMetadataKeys.General.BASE_VERSION, metadata.base_version
        )  # noqa: E202
        metadata.url = metadata_override.get(
            GGUFMetadataKeys.General.URL, metadata.url
        )  # noqa: E202
        metadata.description = metadata_override.get(
            GGUFMetadataKeys.General.DESCRIPTION, metadata.description
        )  # noqa: E202
        metadata.license = metadata_override.get(
            GGUFMetadataKeys.General.LICENSE, metadata.license
        )  # noqa: E202
        metadata.license_name = metadata_override.get(
            GGUFMetadataKeys.General.LICENSE_NAME, metadata.license_name
        )  # noqa: E202
        metadata.license_link = metadata_override.get(
            GGUFMetadataKeys.General.LICENSE_LINK, metadata.license_link
        )  # noqa: E202
        metadata.source_url = metadata_override.get(
            GGUFMetadataKeys.General.SOURCE_URL, metadata.source_url
        )  # noqa: E202
        metadata.source_hf_repo = metadata_override.get(
            GGUFMetadataKeys.General.SOURCE_HF_REPO, metadata.source_hf_repo
        )  # noqa: E202
        metadata.parameter_size_class = metadata_override.get(
            GGUFMetadataKeys.General.PARAMETER_SIZE_CLASS, metadata.parameter_size_class
        )  # noqa: E202
        metadata.tags = metadata_override.get(
            GGUFMetadataKeys.General.TAGS, metadata.tags
        )  # noqa: E202
        metadata.language = metadata_override.get(
            GGUFMetadataKeys.General.LANGUAGE, metadata.language
        )  # noqa: E202
        metadata.datasets = metadata_override.get(
            GGUFMetadataKeys.General.datasets, metadata.datasets
        )  # noqa: E202

        # Direct Metadata Override (via direct cli argument)
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
