#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <sys/wait.h>
#include <wait.h>

void ggml_print_backtrace(void) {
    /*
    #include <dlfcn.h>
    #include <execinfo.h>

    void * trace[100];

    int nptrs = backtrace(trace, sizeof(trace)/sizeof(trace[0]));

    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
    */

    // backtrack_symbols does not show line numbers, use gdb instead
    char attach[32];
    snprintf(attach, sizeof(attach), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        execlp(
            "gdb",
            "gdb",
            "--batch",
            "-ex",
            "set style enabled on",
            "-ex",
            attach,
            "-ex",
            "bt -frame-info source-and-location",
            "-ex",
            "detach",
            "-ex",
            "quit",
            (char*) NULL
        );
    } else {
        waitpid(pid, NULL, 0);
    }
}

#define GGML_ASSERT(x)                                                           \
    do {                                                                         \
        if (!(x)) {                                                              \
            fflush(stdout);                                                      \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ggml_print_backtrace();                                              \
            abort();                                                             \
        }                                                                        \
    } while (0)

enum gguf_vocab_type {
    GGUF_VOCAB_TYPE_NONE = 0, // Models has no vocab
    GGUF_VOCAB_TYPE_SPM  = 1, // Tokenizer Byte Level BPE with byte fallback
    GGUF_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer Byte Level BPE
    GGUF_VOCAB_TYPE_WPM  = 3, // BERT tokenizer Byte Level WordPiece
};

enum gguf_token_type {
    GGUF_TOKEN_TYPE_UNDEFINED    = 0,
    GGUF_TOKEN_TYPE_NORMAL       = 1,
    GGUF_TOKEN_TYPE_UNKNOWN      = 2,
    GGUF_TOKEN_TYPE_CONTROL      = 3,
    GGUF_TOKEN_TYPE_USER_DEFINED = 4,
    GGUF_TOKEN_TYPE_UNUSED       = 5,
    GGUF_TOKEN_TYPE_BYTE         = 6,
};

enum gguf_token_flags {
    GGUF_TOKEN_FLAG_SPECIAL     = 0,
    GGUF_TOKEN_FLAG_NORMALIZED  = 1,
    GGUF_TOKEN_FLAG_LSTRIP      = 2,
    GGUF_TOKEN_FLAG_RSTRIP      = 3,
    GGUF_TOKEN_FLAG_SINGLE_WORD = 4,
};

struct gguf_vocab {
    using id     = int32_t;
    using token  = std::string;
    using ttype  = gguf_token_type;
    using tflags = gguf_token_flags;

    struct token_data {
        token  text;
        float  score;
        ttype  type;
        tflags flags;
    };

    std::string              arch;
    enum gguf_vocab_type     type = GGUF_VOCAB_TYPE_SPM;
    std::vector<std::string> norm;
    std::vector<std::string> pre;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::vector<id> special_tokens_cache;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id  = 1;
    id special_eos_id  = 2;
    id special_unk_id  = 0;
    id special_sep_id  = -1;
    id special_pad_id  = -1;
    id special_cls_id  = -1;
    id special_mask_id = -1;

    int special_add_bos = -1; // -1 unknown, 1 add, 0 don't add.
    int special_add_eos = -1; // -1 unknown, 1 add, 0 don't add.

    id linefeed_id       = 13;
    id special_prefix_id = -1;
    id special_suffix_id = -1;
    id special_middle_id = -1;
    id special_eot_id = -1; // TODO: move above after "eos_id", and here add "file separator" token

    bool add_space_prefix = true;

    int find_bpe_rank(const std::string &token_left, const std::string &token_right) const {
        GGML_ASSERT(token_left.find(' ') == std::string::npos);
        GGML_ASSERT(token_left.find('\n') == std::string::npos);
        GGML_ASSERT(token_right.find(' ') == std::string::npos);
        GGML_ASSERT(token_right.find('\n') == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }
};

int main(int argc, char* argv[]) {
    if (1 == argc) {
        puts("Usage: vocab [-f <file>] [-v <vocab-type>]");
        return 1;
    }

    const char* const   short_options = "f:v:";
    const struct option long_options[]
        = {{"registry-file", required_argument, nullptr, 'f'},
           {"vocab-type", optional_argument, nullptr, 'v'}};

    int  opt;
    char registry_file_path[1024];
    char vocab_name[5] = "SPM"; // NOTE: avoid name conflict

    while ((opt = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                if (optarg == nullptr || strlen(optarg) < 1) {
                    puts("Error: Invalid file path specified.");
                    return 1;
                }

                strcpy(registry_file_path, optarg);
                break;

            case 'v':
                if (optarg == nullptr || strlen(optarg) > 3) {
                    puts("Error: Invalid vocab type specified.");
                    return 1;
                }

                strncpy(vocab_name, optarg, 5);
                break;

            default:
                puts("Usage: vocab [-f <file>] [-v <vocab-type>]");
                return 1;
        }
    }

    std::ifstream  f(registry_file_path);
    nlohmann::json data = nlohmann::json::parse(f);

    if (data.is_null()) {
        puts("Error: Unable to parse registry file.");
        return 1;
    }

    for (const auto &config : data) {
        const std::string model_repo = config["model_repo"];
        const std::string model_arch = config["model_arch"];
        const std::string vocab_type = config["vocab_type"];

        if (vocab_type == vocab_name) {
            std::cout << "Model Repository:\t" << model_repo << '\n';
            std::cout << "Model Architecture:\t" << model_arch << '\n';

            // normalizer and pre_tokenizer are either null or an object
            const nlohmann::json normalizer    = config["normalizer"];
            const nlohmann::json pre_tokenizer = config["pre_tokenizer"];

            // NOTE: Normalizer may be one of null, Sequence, NFC, NFD, NFKC, NFKD...
            // Seems to be null, Sequence, or NFC in most cases
            // Attempt to handle cases where a key may or may not be present and have a null value
            // Not sure whether to default to NFD or NFC if normalizer is null
            // NOTE: The normalizer type key is not guaranteed
            if (!normalizer.is_null()) {
                const std::string norm_type = normalizer["type"];
                std::cout << "Normalizer Type:\t" << norm_type << std::endl;
                // sequence is an array
                if (0 == norm_type.compare("Sequence")) {
                    const nlohmann::json normalizers = normalizer["normalizers"];
                    if (normalizers.is_array()) {
                        for (const auto &norm : normalizers.items()) {
                            std::cout << "Norm Sequence Object:\t" << norm.key() << ":\t"
                                      << norm.value() << std::endl;
                        }
                    }
                } else {
                    // otherwise norm_type is an object potentially containing properties
                    // this varies from model to model
                    // maybe we can just dump the entire object? i don't know yet...
                    for (const auto &norm : normalizer.items()) {
                        std::cout << "Pre Object Pair:\t" << norm.key() << ":\t" << norm.value()
                                  << std::endl;
                    }
                }
            }

            // NOTE: The pre_tokenizer type is not guaranteed
            if (!pre_tokenizer.is_null()) {
                const std::string pre_tokenizer_type = pre_tokenizer["type"];
                std::cout << "Pre-tokenizer Type:" << pre_tokenizer_type << std::endl;
                // sequence is an array
                if (0 == pre_tokenizer_type.compare("Sequence")) {
                    // extract the array
                    // each element in the array is an object
                    const nlohmann::json pretokenizers = pre_tokenizer["pretokenizers"];
                    if (pretokenizers.is_array()) {
                        for (const auto &pre : pretokenizers.items()) {
                            std::cout << "Pre Sequence Object:\t" << pre.key() << ":\t"
                                      << pre.value() << std::endl;
                        }
                    }
                } else {
                    // otherwise pre_tokenizer_type is an object potentially containing properties
                    // this varies from model to model
                    // maybe we can just dump the entire object?
                    // i don't know yet
                    for (const auto &pre : pre_tokenizer.items()) {
                        std::cout << "Pre Object Pair:\t" << pre.key() << ":\t" << pre.value()
                                  << std::endl;
                    }
                }
            }
        }
    }

    return 0;
}
