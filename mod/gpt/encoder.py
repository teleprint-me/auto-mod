import json
import os
import re
from functools import lru_cache
from typing import IO


class Encoder:
    def __init__(
        self,
        encoder: IO,
        bpe_merges: set[tuple[str, str]],
    ):
        self.encoder = encoder
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @staticmethod
    def get_encoder(model_name: str, models_dir: str) -> "Encoder":
        json_encoder_path = os.path.join(models_dir, model_name, "encoder.json")
        with open(json_encoder_path, "r", encoding="utf-8") as f:
            encoder = json.load(f)

        bpe_vocab_path = os.path.join(models_dir, model_name, "vocab.bpe")
        with open(bpe_vocab_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()

        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]
        ]
        return Encoder(encoder, bpe_merges)

    @lru_cache()
    @staticmethod
    def bytes_to_unicode(size: int = 256) -> dict[int, str]:
        """
        This function generates a dictionary mapping each byte to its corresponding Unicode character.

        :param size: The total number of bytes in the encoding space (default is 256 for ASCII).

        :return: A dictionary containing mappings between bytes and their respective Unicode characters.
        """

        # list of visible characters:
        # (ord("!"), ord("~") + 1); (ord("¡"), ord("¬") + 1); (ord("®"), ord("ÿ") + 1);
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))

        mapping: dict = {}
        for byte in list(range(size)):
            # convert "visible" characters
            if byte in visible:
                mapping[byte] = chr(byte)
            else:  # translate and convert non-printable characters
                mapping[byte] = chr(byte + size)
        return mapping

    @staticmethod
    def get_pairs(word: str) -> set[tuple[str, str]]:
        """
        This function

        :param word: A tuple of symbols which are variable-length strings.

        :return A set of tuples representing the paired symbols of a word.
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @property
    def byte_encoder(self) -> dict[int, str]:
        return Encoder.bytes_to_unicode()

    @property
    def byte_decoder(self) -> dict[str, int]:
        return {v: k for k, v in self.byte_encoder.items()}

    @property
    def decoder(self) -> dict:
        return {v: k for k, v in self.encoder.items()}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = Encoder.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = Encoder.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        for token in re.findall(self.pattern, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: list[int], errors: str = "replace") -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=errors  # handle decoding errors
        )
        return text
