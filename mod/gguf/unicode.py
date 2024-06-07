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
import unicodedata

import regex


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
    nfd: list[tuple[int, int]] = dataclasses.field(default_factory=list)


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
    nfd: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)


class CodepointProcessor:
    """
    The `CodepointProcessor` class precomputes various data related to Unicode characters, such as flags, whitespace codes, lowercase and uppercase character ranges, normalized form D (NFD) mappings, etc.,
        which can be used to improve the performance of text processing tasks. This class is primarily useful when working with large amounts of text data that require frequent lookups or manipulations based on Unicode properties,
            as it provides constant-time access to precomputed data instead of having to perform expensive computations at runtime.

    The `CodepointProcessor` can be initialized by specifying the maximum number of code points (Unicode characters) to process. If no limit is provided, all valid Unicode characters will be processed up to U+10FFFF.

    Once instantiated, you should call the `process_unicode()` method to compute and store precomputed data for each character within its defined limits. After processing,
        you can access various properties such as flags, whitespace codes, lowercase/uppercase ranges, normalized form D mappings, etc., using standard Python attribute syntax:

        ```python
        from tok.gguf import unicode

        processor = unicode.CodepointProcessor()
        processor.process_unicode()

        if 9 == processor.unicode_table.whitespace[0]:
            print("The first whitespace code is a tab.")

        lowercase_range = (processor.unicode_table.lowercase[0][0],
                           processor.unicode_table.lowercase[-1][1])

        uppercase_range = (processor.unicode_table.uppercase[0][0],
                           processor.unicode_table.uppercase[-1][1])

        print(f"Lowercase range: {ord('a')} - {lowercase_range[1]}")

        print(f"Uppercase range: {ord('A')} - {uppercase_range[1]}")

        # ...
        ```
    """

    def __init__(self, max_codepoints: None | int = 0x110000):
        # Set the unicode upper limit
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

        # Set the unicode components
        self._codepoint_flags = (CodepointFlags * self.MAX_CODEPOINTS)()
        self._codepoint_ranges = CodepointRanges()
        self._unicode_table = UnicodeTable()

    @property
    def codepoint_flags(self) -> CodepointFlags:
        return self._codepoint_flags

    @property
    def codepoint_ranges(self) -> CodepointRanges:
        return self._codepoint_ranges

    @property
    def unicode_table(self) -> UnicodeTable:
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
        # whitespace
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

    def group_flag_ranges(self):
        # group ranges with same flags
        self._codepoint_ranges.flags = [(0, self._codepoint_flags[0])]  # start, flags
        for codepoint, flags in enumerate(self._codepoint_flags):
            if bytes(flags) != bytes(self._codepoint_ranges.flags[-1][1]):
                self._codepoint_ranges.flags.append((codepoint, flags))
        self._codepoint_ranges.flags.append((self.MAX_CODEPOINTS, CodepointFlags()))

    def group_nfd_ranges(self):
        # group ranges with same nfd
        self._codepoint_ranges.nfd = [(0, 0, 0)]  # start, last, nfd
        for codepoint, norm in self.unicode_table.nfd:
            start = self._codepoint_ranges.nfd[-1][0]
            if self._codepoint_ranges.nfd[-1] != (start, codepoint - 1, norm):
                self._codepoint_ranges.nfd.append(None)
                start = codepoint
            self._codepoint_ranges.nfd[-1] = (start, codepoint, norm)
