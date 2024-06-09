#ifndef AUTO_MOD_MAGIC_H
#define AUTO_MOD_MAGIC_H

#include <stdint.h>
#include <stdio.h>

#define MAGIC_HEADER 4

enum magic_precision_t {
    MAGIC_PRECISION_F32,
    MAGIC_PRECISION_F16,
    MAGIC_PRECISION_Q8_0,
    MAGIC_PRECISION_Q4_0,
    MAGIC_PRECISION_I32,
    MAGIC_PRECISION_I16,
    MAGIC_PRECISION_I8,
    MAGIC_PRECISION_COUNT,
};

enum magic_data_t {
    MAGIC_DATA_TYPE_FLOAT32,
    MAGIC_DATA_TYPE_FLOAT16,
    MAGIC_DATA_TYPE_QUANT8_0,
    MAGIC_DATA_TYPE_QUANT4_0,
    MAGIC_DATA_TYPE_INT32,
    MAGIC_DATA_TYPE_INT16,
    MAGIC_DATA_TYPE_INT8,
    MAGIC_DATA_TYPE_BOOL,
    MAGIC_DATA_TYPE_STRING,
    MAGIC_DATA_TYPE_ARRAY,
    MAGIC_DATA_TYPE_OBJECT,
};

struct magic_string_t {
    unsigned long int length;
    char*             string;
};

union magic_value_t {
    _Float32 float32;
    _Float16 float16;
    int32_t  int32;
    int16_t  int16;
    int8_t   int8;
    bool     boolean;

    struct magic_string_t string;

    struct magic_array_t {
        magic_data_t         type;
        size_t               length;
        union magic_value_t* elements;
    };
};

struct magic_kv_t {
    magic_string_t key;
    magic_data_t   value_type;
    magic_value_t  value;
    size_t         length;
};

struct magic_header_t {
    char         name[MAGIC_HEADER];
    unsigned int value;
    unsigned int version;
};

struct magic_file_t {
    char*                 path;
    FILE*                 file;
    struct magic_header_t header;
};

#endif // AUTO_MOD_MAGIC_H
