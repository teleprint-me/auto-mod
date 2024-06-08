#include <endian.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

#define MAGIC_HEADER 4

struct magic_file {
    char*        path;
    FILE*        file;
    char         header[MAGIC_HEADER];
    unsigned int value;
};

struct options {
    struct magic_file magic;
};

const char* const    short_options  = "f:";
static struct option long_options[] = {{"file", required_argument, 0, 'f'}, {NULL}};

int main(int argc, char* argv[]) {
    if (1 == argc) {
        printf("Usage: %s [-f <file>]\n", argv[0]);
        return 1;
    }

    struct options opts;
    int            c;

    memset(&opts.magic, '\0', sizeof(struct magic_file));

    while (-1 != (c = getopt_long(argc, argv, short_options, long_options, NULL))) {
        switch (c) {
            case 'f':
                opts.magic.path = optarg;
                break;

            default:
                printf("Invalid option: %s\n", optarg);
                break;
        }
    }

    if (NULL == opts.magic.path) {
        printf("Error: File path not provided.\n");
        return 1;
    }

    opts.magic.file = fopen(opts.magic.path, "rb");

    // Open the binary file
    if (NULL == opts.magic.file) {
        printf("Error opening file\n");
        return 1;
    }

    memset(opts.magic.header, '\0', sizeof(opts.magic.header));

    // Read the MAGIC value from the file
    fread(&opts.magic.value, sizeof(unsigned int), 1, opts.magic.file);

    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        // Convert each byte to its corresponding ASCII character
        opts.magic.header[i]
            = opts.magic.value & 0xFF ? ((opts.magic.value >> (8 * i)) & 0xFF) : '\0';
    }

    printf("The magic number is: %#x\n", le32toh(opts.magic.value));

    printf("The magic number as a string: ");

    // Print each character of magic_header
    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        if (opts.magic.header[i] != '\0') {
            putchar(opts.magic.header[i]);
        }
    }

    printf("\n");

    // Close the binary file
    fclose(opts.magic.file);

    return 0;
}
