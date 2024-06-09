#include "magic.h"

#include <endian.h>
#include <getopt.h>
#include <string.h>

const char* const    short_options  = "f:";
static struct option long_options[] = {{"file", required_argument, 0, 'f'}, {NULL}};

int main(int argc, char* argv[]) {
    if (1 == argc) {
        printf("Usage: %s [-f <file>]\n", argv[0]);
        return 1;
    }

    struct magic_file_t magic;
    int                 c;

    memset(&magic, '\0', sizeof(struct magic_file_t));
    memset(&magic.header, '\0', sizeof(struct magic_header_t));

    while (-1 != (c = getopt_long(argc, argv, short_options, long_options, NULL))) {
        switch (c) {
            case 'f':
                magic.path = optarg;
                break;

            default:
                printf("Invalid option: %s\n", optarg);
                break;
        }
    }

    if (NULL == magic.path) {
        printf("Error: File path not provided.\n");
        return 1;
    }

    magic.file = fopen(magic.path, "rb");

    // Open the binary file
    if (NULL == magic.file) {
        printf("Error opening file\n");
        return 1;
    }

    // Read the MAGIC value from the file
    fread(&magic.header.value, sizeof(unsigned int), 1, magic.file);

    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        // Convert each byte to its corresponding ASCII character
        magic.header.name[i]
            = magic.header.value & 0xFF ? ((magic.header.value >> (8 * i)) & 0xFF) : '\0';
    }

    printf("The magic number is: %#x\n", le32toh(magic.header.value));

    printf("The magic number as a string: ");

    // Print each character of magic_header
    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        if (magic.header.name[i] != '\0') {
            putchar(magic.header.name[i]);
        }
    }

    printf("\n");

    // Close the binary file
    fclose(magic.file);

    return 0;
}
