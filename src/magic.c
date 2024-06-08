#include <endian.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

#define MAGIC_HEADER 4

struct magic_file {
    char* path;
    FILE* file;
    char* name[MAGIC_HEADER];
};

struct options {
    struct magic_file magic;
};

int main(int argc, char* argv[]) {
    if (1 == argc) {
        printf("Usage: %s [-f <file>]", argv[0]);
        return 1;
    }

    FILE*        file;
    unsigned int magic;
    char         magic_header[MAGIC_HEADER];

    // Open the binary file
    if ((file = fopen("my_binary_file.bin", "rb")) == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    // Read the MAGIC value from the file
    fread(&magic, sizeof(unsigned int), 1, file);

    memset(magic_header, '\0', sizeof(magic_header));

    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        // Convert each byte to its corresponding ASCII character
        magic_header[i] = magic & 0xFF ? ((magic >> (8 * i)) & 0xFF) : '\0';
    }

    printf("The magic number is: %#x\n", le32toh(magic));

    printf("The magic number as a string: ");

    // Print each character of magic_header
    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        if (magic_header[i] != '\0') {
            putchar(magic_header[i]);
        }
    }

    printf("\n");

    // Close the binary file
    fclose(file);

    return 0;
}
