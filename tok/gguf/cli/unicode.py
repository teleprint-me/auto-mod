import ctypes

from ..unicode import CodepointFlags, CodepointProcessor

# Generate 'unicode-data.cpp'


def main():
    assert ctypes.sizeof(CodepointFlags) == 2

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
