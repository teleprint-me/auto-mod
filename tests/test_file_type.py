import os

import pytest

from tok.file_type import FileType


@pytest.fixture
def file_type() -> FileType:
    return FileType()


def test_bpe_file(file_type):
    assert file_type.get_file_type("tokenizers/bpe/tokenizer.model") == "text/plain"


def test_sentencepiece_model(file_type):
    assert (
        file_type.get_file_type("tokenizers/spm/tokenizer.model") == "application/spm"
    )


def test_json_file(file_type):
    assert file_type.get_file_type("tokenizers/hf/tokenizer.json") == "application/json"


def test_gguf_file(file_type):
    assert (
        file_type.get_file_type("tokenizers/gguf/tokenizer.model") == "application/gguf"
    )


def test_unknown_file(file_type):
    assert file_type.get_file_type("test_unknown_file.txt") is None


def test_nonexistent_file(file_type):
    with pytest.raises(FileNotFoundError):
        file_type.get_file_type("test_nonexistent_file.txt")


def test_plaintext_file(file_type):
    # create test file
    path = "test_plain_file.txt"
    with open(path, "w") as f:
        f.write("This is just a text 😸")

    # test file before cleanup
    assert file_type.get_file_type(path) == "text/plain"

    # cleanup file after test
    os.remove(path)


def test_empty_file(file_type):
    path = "temp_empty_file.txt"
    with open(path, "w") as f:
        f.write("")

    # test file before cleanup
    assert file_type.get_file_type(path) == "text/plain"

    # cleanup file after test
    os.remove(path)
