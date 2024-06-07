from __future__ import annotations

import json
import string
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Optional

import frontmatter

MAGIC = int.from_bytes("AUTO".encode(), byteorder="little", signed=False)
VERSION = 1
DEFAULT_ALIGNMENT = 32
QUANTIZATION_VERSION = 2  # reference ggml.h


class MetadataKeys:
    class General:
        ARCHITECTURE = "general.architecture"
        QUANTIZATION_VERSION = "general.quantization_version"
        ALIGNMENT = "general.alignment"
        NAME = "general.name"
        BASENAME = "general.basename"
        FINETUNE = "general.finetune"
        AUTHOR = "general.author"
        ORGANIZATION = "general.organization"
        VERSION = "general.version"
        BASE_VERSION = "general.base_version"
        URL = "general.url"
        DESCRIPTION = "general.description"
        LICENSE = "general.license"
        LICENSE_NAME = "general.license.name"
        LICENSE_LINK = "general.license.link"
        SOURCE_URL = "general.source.url"
        SOURCE_REPO = "general.source.repository"
        FILE_TYPE = "general.file_type"
        PARAMETER_SIZE_CLASS = "general.parameter_size_class"
        TAGS = "general.tags"
        LANGUAGE = "general.language"
        DATASETS = "general.datasets"
        ENDIANESS = "general.endianess"  # little or big

    class LLM:
        VOCAB_SIZE = "{arch}.vocab_size"
        CONTEXT_LENGTH = "{arch}.context_length"
        EMBEDDING_LENGTH = "{arch}.embedding_length"
        BLOCK_COUNT = "{arch}.block_count"
        LEADING_DENSE_BLOCK_COUNT = "{arch}.leading_dense_block_count"
        FEED_FORWARD_LENGTH = "{arch}.feed_forward_length"
        EXPERT_FEED_FORWARD_LENGTH = "{arch}.expert_feed_forward_length"
        USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
        TENSOR_DATA_LAYOUT = "{arch}.tensor_data_layout"
        EXPERT_COUNT = "{arch}.expert_count"
        EXPERT_USED_COUNT = "{arch}.expert_used_count"
        EXPERT_SHARED_COUNT = "{arch}.expert_shared_count"
        EXPERT_WEIGHTS_SCALE = "{arch}.expert_weights_scale"
        POOLING_TYPE = "{arch}.pooling_type"
        LOGIT_SCALE = "{arch}.logit_scale"

    class Attention:
        HEAD_COUNT = "{arch}.attention.head_count"
        HEAD_COUNT_KV = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV = "{arch}.attention.clamp_kqv"
        KEY_LENGTH = "{arch}.attention.key_length"
        VALUE_LENGTH = "{arch}.attention.value_length"
        LAYERNORM_EPS = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"
        CAUSAL = "{arch}.attention.causal"
        Q_LORA_RANK = "{arch}.attention.q_lora_rank"
        KV_LORA_RANK = "{arch}.attention.kv_lora_rank"

    class Rope:
        DIMENSION_COUNT = "{arch}.rope.dimension_count"
        FREQ_BASE = "{arch}.rope.freq_base"
        SCALING_TYPE = "{arch}.rope.scaling.type"
        SCALING_FACTOR = "{arch}.rope.scaling.factor"
        SCALING_ATTN_FACTOR = "{arch}.rope.scaling.attn_factor"
        SCALING_ORIG_CTX_LEN = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED = "{arch}.rope.scaling.finetuned"
        SCALING_YARN_LOG_MUL = "{arch}.rope.scaling.yarn_log_multiplier"

    class SSM:
        CONV_KERNEL = "{arch}.ssm.conv_kernel"
        INNER_SIZE = "{arch}.ssm.inner_size"
        STATE_SIZE = "{arch}.ssm.state_size"
        TIME_STEP_RANK = "{arch}.ssm.time_step_rank"

    class Tokenizer:
        MODEL = "tokenizer.model"  # STRING: e.g. llama, gpt2, etc...
        TYPE = "tokenizer.type"  # STRING: BPE, SPM, WPM, etc.
        NORM = "tokenizer.norm"  # OBJECT {"type": "ByteLevel", ...}
        PRE = "tokenizer.pre"  # OBJECT {"type": "ByteLevel", ...}
        ADDED = "tokenizer.added"  # ARRAY of OBJECTs: [{"id": 1, ...}, ...]
        VOCAB = "tokenizer.vocab"  # ARRAY of STRINGs: ["[BOS]", ...]
        MERGES = "tokenizer.merges"  # ARRAY of STRINGs: ["▁ t", ...]
        TOKEN_TYPE = "tokenizer.token_type"  # ARRAY of INT [2, ...]
        TOKEN_TYPE_COUNT = "tokenizer.token_type_count"  # BERT token types
        SCORES = "tokenizer.scores"  # WPM only
        BOS_ID = "tokenizer.bos_token_id"
        EOS_ID = "tokenizer.eos_token_id"
        UNK_ID = "tokenizer.unknown_token_id"
        SEP_ID = "tokenizer.separator_token_id"
        PAD_ID = "tokenizer.padding_token_id"
        CLS_ID = "tokenizer.cls_token_id"
        MASK_ID = "tokenizer.mask_token_id"
        ADD_BOS = "tokenizer.add_bos_token"
        ADD_EOS = "tokenizer.add_eos_token"
        ADD_PREFIX = "tokenizer.add_space_prefix"
        RWKV = "tokenizer.rwkv.world"
        CHAT_TEMPLATE = "tokenizer.chat_template"
        CHAT_TEMPLATE_N = "tokenizer.chat_template.{name}"
        CHAT_TEMPLATES = "tokenizer.chat_templates"
        # FIM/Infill special tokens constants
        PREFIX_ID = "tokenizer.prefix_token_id"
        SUFFIX_ID = "tokenizer.suffix_token_id"
        MIDDLE_ID = "tokenizer.middle_token_id"
        EOT_ID = "tokenizer.eot_token_id"


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
    ) -> Metadata:
        # This grabs as many contextual authorship metadata as possible from the model repository
        # making any conversion as required to match the gguf kv store metadata format
        # as well as giving users the ability to override any authorship metadata that may be incorrect

        # Create a new Metadata instance
        metadata = Metadata()

        # load model folder model card if available
        # Reference Model Card Metadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
        model_card = Metadata.load_model_card(model_path)

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
        hf_params = Metadata.load_huggingface_parameters(model_path)
        hf_name_or_path = hf_params.get("_name_or_path")
        if metadata.name is None and hf_name_or_path is not None:
            metadata.name = Path(hf_name_or_path).name
        if metadata.source_repo is None and hf_name_or_path is not None:
            metadata.source_repo = Path(hf_name_or_path).name

        # Use Directory Folder Name As Fallback Name
        if metadata.name is None and model_path is not None and model_path.exists():
            metadata.name = model_path.name

        # Metadata Override File Provided
        if metadata_override_path is not None:
            override = Metadata.load_metadata_override(metadata_override_path)
            # metadata.<attr> = override.get(Keys.General.<attr>, metadata.<attr>)
            for k, v in vars(GGUFMetadataKeys.General).items():
                if not k.startswith("__"):
                    key = k.lower()
                    val = override.get(v, getattr(metadata, key))
                    setattr(metadata, key, val)

        # Direct Metadata Override (via direct cli argument)
        if model_name is not None:
            metadata.name = model_name

        return metadata

    @staticmethod
    def _load_file(path: Optional[Path] = None) -> dict[str, object]:
        if path is None or not path.exists():
            return dict()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if path.suffix in [".md", ".yaml", ".yml"]:
            # Model card file
            return frontmatter.load(content)

        elif path.suffix == ".json":
            # Other files (like config.json, metadata_override.json, etc.)
            return json.loads(content)

        else:
            raise ValueError("Unsupported file type.")

    @staticmethod
    def load_metadata_override(
        metadata_override_path: Optional[Path] = None,
    ) -> dict[str, object]:
        return Metadata._load_file(metadata_override_path)

    @staticmethod
    def load_model_card(
        model_path: Optional[Path] = None,
    ) -> dict[str, object]:
        return Metadata._load_file(model_path / "README.md")

    @staticmethod
    def load_huggingface_parameters(
        model_path: Optional[Path] = None,
    ) -> dict[str, object]:
        return Metadata._load_file(model_path / "config.json")
