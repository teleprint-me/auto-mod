"""
Module: tok.gguf.cli.unicode
"""

import argparse
import ctypes
from pathlib import Path

from ..unicode import CodepointFlags, CodepointProcessor

# Generate 'unicode-data.cpp'


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 'unicode-data.cpp'")

    # output path - default to stdout if no output path is given
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output file path (default: stdout)",
    )

    # endianess - default to little-endian if no option provided
    parser.add_argument(
        "--big-endian",
        action="store_true",
        help="The byte order of the code points (default: False ('little'))",
    )

    # max_codepoints - default to 0x110000 if no boundary is given
    parser.add_argument(
        "--max-codepoints",
        type=int,
        default=0x110000,
        help="Maximum code points limit (default: 0x110000)",
    )

    return parser.parse_args()


# TODO: define helper functions for setting mapping?


def build_unicode_source(processor: CodepointProcessor) -> str:
    # define includes
    # set ranges flags
    # set whitespace
    # map lowercase
    # map uppercase
    # set ranges nfd
    pass


def main():
    assert ctypes.sizeof(CodepointFlags) == 2

    args = get_arguments()

    codepoint_processor = CodepointProcessor(0x110000)

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
    for codepoint, flags in codepoint_processor.codepoint_ranges.flags:
        flags = int.from_bytes(bytes(flags), "little")
        out("{0x%06X, 0x%04X}," % (codepoint, flags))
    out("};\n")

    out("const std::unordered_set<uint32_t> unicode_set_whitespace = {")
    out(
        ", ".join(
            "0x%06X" % cpt for cpt in codepoint_processor.unicode_table.whitespace
        )
    )
    out("};\n")

    out("const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {")
    for tuple in codepoint_processor.unicode_table.lowercase:
        out("{0x%06X, 0x%06X}," % tuple)
    out("};\n")

    out("const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {")
    for tuple in codepoint_processor.unicode_table.uppercase:
        out("{0x%06X, 0x%06X}," % tuple)
    out("};\n")

    out("const std::vector<range_nfd> unicode_ranges_nfd = {  // start, last, nfd")
    for triple in codepoint_processor.codepoint_ranges.nfd:
        out("{0x%06X, 0x%06X, 0x%06X}," % triple)
    out("};\n")


if __name__ == "__main__":
    main()
