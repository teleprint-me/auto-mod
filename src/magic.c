#include <endian.h>
#include <stdio.h>
#include <string.h>

#define MAGIC_HEADER 4

int main(void) {
    FILE*        file;
    unsigned int magic;

    // Open the binary file
    if ((file = fopen("my_binary_file.bin", "rb")) == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    // Read the MAGIC value from the file
    fread(&magic, sizeof(unsigned int), 1, file);

    char hex_str[MAGIC_HEADER];

    memset(hex_str, '\0', sizeof(hex_str));

    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        // Convert each byte to its corresponding ASCII character
        hex_str[i] = magic & 0xFF ? ((magic >> (8 * i)) & 0xFF) : '\0';
    }

    printf("The magic number is: %#x\n", le32toh(magic));

    printf("The magic number as a string: ");

    // Print each character of hex_str
    for (size_t i = 0; i < MAGIC_HEADER; ++i) {
        if (hex_str[i] != '\0') {
            putchar(hex_str[i]);
        }
    }

    printf("\n");

    // Close the binary file
    fclose(file);

    return 0;
}
