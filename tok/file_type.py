import argparse
import json


class FileType:
    @staticmethod
    def is_binary(file_name: str) -> bool:
        """
        Determines if a given file is a binary tokenizer model.

        This method exploits the fact that attempting to read a binary file
        using the default 'r' mode raises a UnicodeDecodeError exception,
        allowing us to make an informed decision based on the observed byte
        sequence.

        :param file_name: The path to the file to be checked.
        :return: A boolean value indicating whether the given file is a binary
                 SentencePiece tokenizer model or not.

        :citation: https://stackoverflow.com/a/51495076/15147156
        """

        try:
            # Attempt to read the file as plaintext using default 'r' mode
            with open(file_name, "r") as file:
                file.read()

                # If we reached this point, it means the file is not a
                # binary file and raises a UnicodeDecodeError exception
                # when attempting to read it in text mode
                return False

        except UnicodeDecodeError:
            with open(file_name, "rb") as file:
                header = file.read(4)

                # Check for the specific byte sequence associated with
                # SentencePiece models
                if header == b"\n\x0e\n\x05":
                    return True

                # GGUF models
                if header == b'GGUF':  # little endian
                    return True

                # If the file is not a binary SentencePiece model, raise an exception
                raise LookupError("Unrecognized binary tokenizer model format")

    @staticmethod
    def is_json(file_name: str) -> bool:
        """
        Determines if a given file is a JSON tokenizer file.

        This method checks for the presence of a "model" key in the JSON data
        loaded from the file, which is a common structure for Hugging Face
        tokenizer configuration files. This method can be extended for other
        implementations as well.

        :param file_name: The path to the file to be checked.
        :return: A boolean value indicating whether the given file is a
        tokenizer JSON file or not.

        :raises: json.JSONDecodeError if the file cannot be read as JSON.
        """

        try:  # Hack to "short-circuit" complex filetype associations
            with open(file_name, "r") as check_file:
                data = json.load(check_file)

                # Check for the presence of a "model" key in the loaded JSON data
                if "model" in data:
                    return True

                raise LookupError("Unrecognized json tokenizer model format")

        except json.JSONDecodeError:
            return False  # Not a valid JSON file

    def is_plaintext(self, file_name: str) -> bool:
        return not self.is_binary(file_name)

    def get_file_type(self, file_name: str) -> str:
        try:
            if self.is_binary(file_name):
                return "binary"
            # check if json first because json is plaintext too
            elif self.is_json(file_name):
                return "json"
            # check if plaintext if not json
            elif self.is_plaintext(file_name):
                return "plaintext"
            else:  # we don't know what the file type is
                return "unknown"
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
