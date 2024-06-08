#ifndef AUTO_MOD_MAGIC_H
#define AUTO_MOD_MAGIC_H

#include <stdint.h>
#include <stdio.h>

#define MAGIC_HEADER 4

typedef enum magic_file_type {
    MAGIC_FILE_TYPE_F32,
    MAGIC_FILE_TYPE_F16,
    MAGIC_FILE_TYPE_Q8_0,
    MAGIC_FILE_TYPE_Q4_0,
    MAGIC_FILE_TYPE_I32,
    MAGIC_FILE_TYPE_I16,
    MAGIC_FILE_TYPE_I8,
    MAGIC_FILE_TYPE_COUNT,
} magic_file_t;

typedef enum magic_value_type {
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
} magic_value_t;

typedef struct magic_string_type {
    unsigned long int length;
    char*             string;
} magic_string_t;

typedef struct magic_array_type {
    magic_value_t type;
    size_t        length;
    magic_data_t* array;
} magic_array_t;

typedef union magic_data_type {
    float          float32;
    _Float16       float16;
    int32_t        int32;
    int16_t        int16;
    int8_t         int8;
    bool           boolean;
    magic_string_t string;
} magic_data_t;

typedef struct magic_header_type {
    unsigned int value;
    unsigned int version;
    unsigned int tensor_count;
} magic_header_t;

struct magic_file {
    char*        path;
    FILE*        file;
    char         header[MAGIC_HEADER];
    unsigned int value;
};

struct options {
    struct magic_file magic;
};

#endif // AUTO_MOD_MAGIC_H
