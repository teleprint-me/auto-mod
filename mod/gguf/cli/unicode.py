"""
Module: tok.gguf.cli.unicode

Generate 'unicode-data.cpp' and 'unicode-data.h'
"""

import argparse
import ctypes
import logging

from ..unicode import CodepointFlags, CodepointProcessor

logger = logging.getLogger(__file__)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 'unicode-data.cpp' and 'unicode-data.h'"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output generated source text (default: False)",
    )

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


def build_unicode_data_h(max_codepoints: int = 0x110000) -> str:
    # NOTE: The resulting string is segmented to prevent formatting conflicts with braces
    unicode_data_h = """\
    // generated with python gguf.cli.unicode
    #ifndef UNICODE_DATA_H
    #define UNICODE_DATA_H

    #include <cstdint>
    #include <vector>
    #include <unordered_map>
    #include <unordered_set>

    /**
     * @brief Represents a Unicode character range with normalized form D (NFD)
     */
    struct range_nfd {
        uint32_t first;
        uint32_t last;
        uint32_t nfd;
    };\n
    """

    unicode_data_h += f"""\
    static const uint32_t MAX_CODEPOINTS = {max_codepoints};\n
    """

    unicode_data_h += """\
    /**
     * @brief Externally linked variables for Unicode data structures
     */
    extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
    extern const std::unordered_set<uint32_t> unicode_set_whitespace;
    extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;
    extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;
    extern const std::vector<range_nfd> unicode_ranges_nfd;
    #endif // UNICODE_DATA_H
    """

    # NOTE: Format source text by line
    return "\n".join([line.strip() for line in unicode_data_h.split("\n")])


# TODO: define helper functions for setting mapping?
def set_ranges_flags(processor: CodepointProcessor, byte_order: str = "little") -> str:
    unicode_ranges_flags = (
        "// codepoint, flag // last=next_start-1\n"
        "const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {\n"
    )
    logger.debug(unicode_ranges_flags)

    for codepoint, flags in processor.codepoint_ranges.flags:
        flags = int.from_bytes(bytes(flags), byte_order)
        line = "{0x%06X, 0x%04X}," % (codepoint, flags)
        logger.debug(line)
        unicode_ranges_flags += line

    line = "};\n\n"
    logger.debug(line)

    return unicode_ranges_flags + line


def set_unicode_whitespace(processor: CodepointProcessor) -> str:
    unicode_set_whitespace = (
        "const std::unordered_set<uint32_t> unicode_set_whitespace = {\n"
    )
    logger.debug(unicode_set_whitespace)

    for codepoint in processor.unicode_table.whitespace:
        line = "0x%06X" % codepoint
        logger.debug(line)
        unicode_set_whitespace += f"{line}, "

    line = "};\n\n"
    logger.debug(line)

    return unicode_set_whitespace + line


def set_unicode_lowercase(processor: CodepointProcessor) -> str:
    unicode_map_lowercase = (
        "const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {\n"
    )

    for tuple in processor.unicode_table.lowercase:
        line = "{0x%06X, 0x%06X}," % tuple
        logger.debug(line)
        unicode_map_lowercase += line

    line = "};\n\n"
    logger.debug(line)

    return unicode_map_lowercase + line


def set_unicode_uppercase(processor: CodepointProcessor) -> str:
    unicode_map_uppercase = (
        "const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {\n"
    )

    for tuple in processor.unicode_table.uppercase:
        line = "{0x%06X, 0x%06X}," % tuple
        logger.debug(line)
        unicode_map_uppercase += line

    line = "};\n\n"
    logger.debug(line)

    return unicode_map_uppercase + line


def set_ranges_nfd(processor: CodepointProcessor) -> str:
    unicode_ranges_nfd = (
        "// start, last, nfd\n" "const std::vector<range_nfd> unicode_ranges_nfd = {\n"
    )

    for triple in processor.codepoint_ranges.nfd:
        line = "{0x%06X, 0x%06X, 0x%06X}," % triple
        logger.debug(line)
        unicode_ranges_nfd += line

    line = "};\n"
    logger.debug(line)

    return unicode_ranges_nfd + line


def build_unicode_data_cpp(processor: CodepointProcessor) -> str:
    # define includes
    unicode_data_cpp = """
    // generated with python gguf.cli.unicode

    #include "unicode-data.h"

    #include <cstdint>
    #include <vector>
    #include <unordered_map>
    #include <unordered_set>\n
    """
    logger.debug(unicode_data_cpp)

    unicode_data_cpp += set_ranges_flags(processor)
    unicode_data_cpp += set_unicode_whitespace(processor)
    unicode_data_cpp += set_unicode_lowercase(processor)
    unicode_data_cpp += set_unicode_uppercase(processor)
    unicode_data_cpp += set_ranges_nfd(processor)

    return unicode_data_cpp


def main():
    assert ctypes.sizeof(CodepointFlags) == 2

    args = get_arguments()

    if args.verbose or not args.output_path:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    processor = CodepointProcessor(args.max_codepoints)
    processor.process_unicode()
    processor.group_flag_ranges()
    processor.group_nfd_ranges()

    # build the header file
    unicode_data_h = build_unicode_data_h(args.max_codepoints)

    # build the source file
    unicode_data_cpp = build_unicode_data_cpp(processor)

    if args.output_path:
        header_file = f"{args.output_path}/unicode-data.h"
        cpp_file = f"{args.output_path}/unicode-data.cpp"

        with open(header_file, "w") as f:
            f.write(unicode_data_h)

        with open(cpp_file, "w") as f:
            f.write(unicode_data_cpp)


if __name__ == "__main__":
    main()
