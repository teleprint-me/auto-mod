"""
Module: tok.gguf.unicode

References:
- Unicode Chapter 3 Unicode Conformance
    - https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf
- Properties accessible through \\p{} and \\P{}
    - https://perldoc.perl.org/perluniprops
- Ctypes
    - https://docs.python.org/3/library/ctypes.html
"""

import ctypes
import dataclasses


class CodepointFlags(ctypes.Structure):
    """
    Represents Unicode character properties as defined by the Unicode Technical Standard #36 (Unicode 5.2) using Python's ctypes library,
        providing boolean flags for various categories of characters based on their unicode values and properties.

    This class allows developers to easily check if a given code point belongs to specific character categories such as numbers (\\p{N}),
        letters (\\p{L}), separators (\\p{Z}), accent marks (\\p{M}), punctuation (\\p{P}), symbols (\\p{S}), and controls (\\p{C}).

    The `CodepointFlags` class uses a structure defined in the unicode.h header file to store these properties efficiently,
        making it suitable for high-performance applications that need to process large amounts of text data with Unicode support.

    To use this class, create an instance of CodepointFlags and call its `from_codepoints` method passing a list or iterable containing the code points
        you want to check, e.g.:

        ```python
        from tok.gguf import unicode

        flags = unicode.CodepointFlags()
        flagged_chars = [0x2145, 0x65, 0xFFFD]

        flags.from_codepoints(flagged_chars)

        for codepoint in flagged_chars:
            print(f"{codepoint}: is_number={flags.is_number(codepoint)}, is_letter={flags.is_letter(codepoint)}")
        ```
    """

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


@dataclasses.dataclass
class UnicodeTable:
    """
    The `UnicodeTable` class serves as a container for various precomputed data related to Unicode characters, such as whitespace codes, lowercase and uppercase character ranges,
        normalized form D (NFD) mappings, etc., which can be used to improve the performance of text processing tasks.

    This class is primarily useful when working with large amounts of text data that require frequent lookups or manipulations based on Unicode properties,
        as it provides constant-time access to precomputed data instead of having to perform expensive computations at runtime.

    The `UnicodeTable` class can be initialized with empty lists for each property (whitespace, lowercase, uppercase, and nfd),
        but the recommended way is to load the necessary data from external files or databases during initialization to ensure accurate and up-to-date information.

    Here's an example of how you can create a `UnicodeTable` instance:

        ```python
        from tok.gguf import unicode

        table = unicode.UnicodeTable()

        # Load data for each property
        with open("whitespace_codes.txt", "r") as f:
            whitespaces = [int(line) for line in f]
            table.whitespace = whitespaces

        # ... continue loading other properties from external files or databases

        ```

    Once the `UnicodeTable` instance is initialized, you can access its properties using standard Python attribute syntax:

        ```python
        if 9 == table.whitespace[0]:
            print("The first whitespace code is a tab.")

        lowercase_range = (table.lowercase[0][0], table.lowercase[-1][1])
        print(f"Lowercase range: {ord('a')} - {lowercase_range[1]}")

        # ...

        ```
    """

    whitespace: list[int] = dataclasses.field(default_factory=list)
    lowercase: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    uppercase: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    nfd: list[tuple[int, int]] = dataclasses.field(default_factor=list)


@dataclasses.dataclass
class CodepointRanges:
    """
    The `CodepointRanges` class serves as a container for precomputed character ranges based on specific Unicode properties, such as character flags and normalized form D (NFD) mappings.

    This class is useful when working with large amounts of text data that require frequent lookups or manipulations based on Unicode properties,
        as it provides constant-time access to precomputed ranges instead of having to perform expensive computations at runtime.

    The `CodepointRanges` can be initialized with empty lists for each property (flags and nfd),
        but the recommended way is to load the necessary data from external files or databases during initialization to ensure accurate and up-to-date information.

    Here's an example of how you can create a `CodepointRanges` instance:

        ```python
        from tok.gguf import unicode

        ranges = unicode.CodepointRanges()

        # Load data for each property
        with open("flags_ranges.txt", "r") as f:
            flagged_codes = [tuple(map(int, line.split("-"))) for line in f]
            ranges.flags = flagged_codes

        # ... continue loading other properties from external files or databases

        ```

    Once the `CodepointRanges` instance is initialized, you can access its properties using standard Python attribute syntax:

        ```python
        for range in ranges.flags:
            start, end = range
            print(f"Flagged character range {start} - {end}")

        # ...

        ```
    """

    flags: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    nfd: list[tuple[int, int, int]] = dataclasses.field(default_factor=list)
