import argparse
import json


class FileType:
    BIN = {
        b"\n\x0e\n\x05": "application/spm",  # not guaranteed
        b"GGUF": "application/gguf",  # ggml/llama.cpp; little endian
        b"PK\x03\x04": "application/zip",  # TODO: pytorch/other; compressed
    }

    @staticmethod
    def _read_json(file_name: str) -> None | str:
        with open(file_name, "r", encoding="utf-8") as check_file:
            data = json.load(check_file)

            # Check for the presence of a "model" key in the loaded JSON data
            if "model" in data:
                return "application/json"

        raise LookupError("Unrecognized json model format")

    @staticmethod
    def _read_text(file_name: str) -> None | str:
        # Attempt to read the file as plaintext using default 'r' mode
        with open(file_name, "r", encoding="utf-8") as file:
            file.read()

            # If we reached this point, it means the file is not a
            # binary file and raises a UnicodeDecodeError exception
            # when attempting to read it in text mode
            return "text/plain"

        raise LookupError("Unrecognized plaintext model format")

    @staticmethod
    def _read_binary(file_name: str) -> None | str:
        with open(file_name, "rb") as file:
            header = file.read(4)

            if len(header) == 0 or header not in FileType.BIN.keys():
                return None

            return FileType.BIN[header]

        raise LookupError("Unrecognized binary model format")

    @staticmethod
    def get_type(file_name: str) -> str:
        try:
            return FileType._read_json(file_name)

        except json.JSONDecodeError:
            return FileType._read_text(file_name)

        except UnicodeDecodeError:
            return FileType._read_binary(file_name)

        except (IOError, OSError) as e:
            raise FileNotFoundError(f"File '{file_name}' not found.") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file-path",
        default="tokenizers/bpe/tokenizer.model",
        help="The input tokenizer model filepath",
    )
    args = parser.parse_args()
    vocab = FileType()
    print(vocab.get_type(args.file_path))


if __name__ == "__main__":
    main()
