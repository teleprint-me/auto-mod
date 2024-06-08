#ifndef AUTO_MOD_MAGIC_H
#define AUTO_MOD_MAGIC_H

#include <stdint.h>
#include <stdio.h>

#define MAGIC_HEADER 4

enum magic_file_t {
    MAGIC_FILE_TYPE_F32,
    MAGIC_FILE_TYPE_F16,
    MAGIC_FILE_TYPE_Q8_0,
    MAGIC_FILE_TYPE_Q4_0,
    MAGIC_FILE_TYPE_I32,
    MAGIC_FILE_TYPE_I16,
    MAGIC_FILE_TYPE_I8,
    MAGIC_FILE_TYPE_COUNT,
};

enum magic_value_t {
    MAGIC_VALUE_TYPE_FLOAT32,
    MAGIC_VALUE_TYPE_FLOAT16,
    MAGIC_VALUE_TYPE_QUANT8_0,
    MAGIC_VALUE_TYPE_QUANT4_0,
    MAGIC_VALUE_TYPE_INT32,
    MAGIC_VALUE_TYPE_INT16,
    MAGIC_VALUE_TYPE_INT8,
    MAGIC_VALUE_TYPE_BOOL,
    MAGIC_VALUE_TYPE_STRING,
    MAGIC_VALUE_TYPE_ARRAY,
    MAGIC_VALUE_TYPE_OBJECT,
};

struct magic_string_t {
    unsigned long int length;
    char*             string;
};

union magic_data_t {
    float                 float32;
    _Float16              float16;
    int32_t               int32;
    int16_t               int16;
    int8_t                int8;
    bool                  boolean;
    struct magic_string_t string;
};

struct magic_array_t {
    magic_value_t       type;
    size_t              length;
    union magic_data_t* array;
};

struct magic_header_t {
    char         header[MAGIC_HEADER];
    unsigned int value;
    unsigned int version;
    unsigned int tensor_count;
};

struct magic_file {
    char*                  path;
    FILE*                  file;
    struct magic_header_t* header;
};

#endif // AUTO_MOD_MAGIC_H
