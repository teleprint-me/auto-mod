import ctypes
import unicodedata

import regex
import dataclasses


# NOTE: Original script has a typo, e.g. CoodepointFlags
class CodepointFlags(ctypes.Structure):
    _fields_ = [  # see definition in unicode.h
        ("is_undefined", ctypes.c_uint16, 1),
        ("is_number", ctypes.c_uint16, 1),  # regex: \p{N}
        ("is_letter", ctypes.c_uint16, 1),  # regex: \p{L}
        ("is_separator", ctypes.c_uint16, 1),  # regex: \p{Z}
        ("is_accent_mark", ctypes.c_uint16, 1),  # regex: \p{M}
        ("is_punctuation", ctypes.c_uint16, 1),  # regex: \p{P}
        ("is_symbol", ctypes.c_uint16, 1),  # regex: \p{S}
        ("is_control", ctypes.c_uint16, 1),  # regex: \p{C}
    ]


assert ctypes.sizeof(CodepointFlags) == 2


@dataclasses.dataclass
class UnicodeTable:
    whitespace: list[int] = []
    lowercase: list[tuple[int, int]] = []
    uppercase: list[tuple[int, int]] = []
    nfd: list[tuple[int, int]] = []


class CodepointProcessor:
    def __init__(self, max_codepoints: None | int = 0x110000):
        self.MAX_CODEPOINTS = max_codepoints
        # Regular expressions for various Unicode character categories
        self._regexes = {
            "is_number": regex.compile(r"\p{N}"),
            "is_letter": regex.compile(r"\p{L}"),
            "is_separator": regex.compile(r"\p{Z}"),
            "is_accent_mark": regex.compile(r"\p{M}"),
            "is_punctuation": regex.compile(r"\p{P}"),
            "is_symbol": regex.compile(r"\p{S}"),
            "is_control": regex.compile(r"\p{C}"),
            "is_whitespace": regex.compile(r"\s"),
        }

        self._codepoint_flags = (CodepointFlags * self.MAX_CODEPOINTS)()

        self._unicode_table = UnicodeTable()

    @property
    def flags(self) -> CodepointFlags:
        return self._codepoint_flags

    @property
    def table(self) -> UnicodeTable:
        return self._unicode_table

    def process_unicode(self):
        for codepoint in range(self.MAX_CODEPOINTS):
            # convert codepoint to unicode character
            char = chr(codepoint)

            # regex categories
            flags = self._codepoint_flags[codepoint]
            for flag in dir(flags):
                if flag.startswith("__"):
                    continue

                regex = self._regexes.get(flag)
                if regex is not None:
                    setattr(flags, flag, bool(regex.match(char)))
                elif flag == "is_undefined":
                    setattr(flags, flag, bytes(flags)[0] == 0)
                    message = f"'flags' is undefined for {codepoint} with {char}"
                    assert not flags.is_undefined, message

            self.set_whitespace_table(codepoint, char)
            self.set_lowercase_table(codepoint, char)
            self.set_uppercase_table(codepoint, char)
            self.set_nfd_table(codepoint, char)

    def set_whitespace_table(self, codepoint: int, char: str):
        regex = self._regexes["is_whitespace"]
        if bool(regex.match(char)):
            self._unicode_table.whitespace.append(codepoint)

    def set_lowercase_table(self, codepoint: int, char: str):
        # lowercase conversion
        lower = ord(char.lower()[0])
        if codepoint != lower:
            self._unicode_table.lowercase.append((codepoint, lower))

    def set_uppercase_table(self, codepoint: int, char: str):
        # uppercase conversion
        upper = ord(char.upper()[0])
        if codepoint != upper:
            self._unicode_table.uppercase.append((codepoint, upper))

    def set_nfd_table(self, codepoint: int, char: str):
        # NFD normalization
        norm = ord(unicodedata.normalize("NFD", char)[0])
        if codepoint != norm:
            self._unicode_table.nfd.append((codepoint, norm))


# group ranges with same flags
ranges_flags = [(0, codepoint_flags[0])]  # start, flags
for codepoint, flags in enumerate(codepoint_flags):
    if bytes(flags) != bytes(ranges_flags[-1][1]):
        ranges_flags.append((codepoint, flags))
ranges_flags.append((MAX_CODEPOINTS, CodepointFlags()))

# group ranges with same nfd
ranges_nfd = [(0, 0, 0)]  # start, last, nfd
for codepoint, norm in table_nfd:
    start = ranges_nfd[-1][0]
    if ranges_nfd[-1] != (start, codepoint - 1, norm):
        ranges_nfd.append(None)
        start = codepoint
    ranges_nfd[-1] = (start, codepoint, norm)


# Generate 'unicode-data.cpp'


def out(line=""):
    print(line, end="\n")  # noqa


out(
    """\
// generated with scripts/gen-unicode-data.py

#include "unicode-data.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
"""
)

out(
    "const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {  // start, flags // last=next_start-1"
)
for codepoint, flags in ranges_flags:
    flags = int.from_bytes(bytes(flags), "little")
    out("{0x%06X, 0x%04X}," % (codepoint, flags))
out("};\n")

out("const std::unordered_set<uint32_t> unicode_set_whitespace = {")
out(", ".join("0x%06X" % cpt for cpt in table_whitespace))
out("};\n")

out("const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {")
for tuple in table_lowercase:
    out("{0x%06X, 0x%06X}," % tuple)
out("};\n")

out("const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {")
for tuple in table_uppercase:
    out("{0x%06X, 0x%06X}," % tuple)
out("};\n")

out("const std::vector<range_nfd> unicode_ranges_nfd = {  // start, last, nfd")
for triple in ranges_nfd:
    out("{0x%06X, 0x%06X, 0x%06X}," % triple)
out("};\n")