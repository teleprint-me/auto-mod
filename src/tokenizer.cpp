#include <climits>
#include <cstdarg>
#include <cstring>
#include <format>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <string>
#include <sys/wait.h>
#include <vector>
#include <wait.h>

// NOTE: MetaToken represents an object element found within added_tokens in the tokenizer.json
struct MetaToken {
    // NOTE: These are required to be defined at runtime.
    int         id;
    std::string content;

    // NOTE: Always default to false. Never assume anything is true.
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

// TODO/WIP
struct Normalizer {
    std::string type; // type is always available if normalizer is not null

    // TODO: Handle sequences if type is Sequence
    std::vector<nlohmann::json> rules;

    // default to false because these member attributes may not be available
    bool add_prefix_space = false;
    bool trim_offsets     = false;
};

// TODO/WIP
struct PreTokenizer {
    std::string type; // type is always available if pre_tokenizer is not null

    // TODO: Handle sequences if type is Sequence
    std::vector<nlohmann::json> rules;

    // default to false because these member attributes may not be available
    bool add_prefix_space = false;
    bool trim_offsets     = false;
};

struct Vocabulary {
  public:
    size_t                        size;
    std::map<std::string, size_t> map;
    std::vector<std::string>      tokens;

    std::vector<struct MetaToken*> added_tokens;

    // Need to know these advance. Must be set on a model-by-model basis as a result.
    struct MetaToken* bos_token = nullptr;
    struct MetaToken* eos_token = nullptr;
    struct MetaToken* unk_token = nullptr;

    std::vector<std::string> merges;

    Vocabulary(const nlohmann::json &model) {
        // model is the metadata as a JSON object
        size   = model["vocab"].size(); // size of the vocab
        map    = model["vocab"];        // vocab is a mapping between tokens and ids
        merges = model["merges"];       // merges is a vector of strings

        for (const auto &pair : model["vocab"].items()) {
            // invert the mapping
            std::string token = pair.key();
            size_t      id    = pair.value();
            // map the vector element
            tokens[id]        = token;
        }
    }
};

struct TokenizerModel {

    std::string type;

    // Encapsulate the vocabulary and related items together
    Vocabulary vocab;

    size_t size() {
        return vocab.size;
    };

    size_t token_to_id(const std::string &token) {
        return vocab.map[token];
    };

    std::string id_to_token(size_t encoding) const {
        return vocab.tokens[encoding];
    };

    bool fuse_unk      = false;
    bool byte_fallback = false;

    // TODO/WIP: Note that normalize and pre_tokenizer are variable objects
    nlohmann::json normalizer;
    nlohmann::json pre_tokenizer;
};

std::vector<struct MetaToken*> allocate_meta_tokens(nlohmann::json added_tokens) {

    if (added_tokens.is_null()) {
        throw std::invalid_argument("Expected a valid added_tokens argument, got null instead.");
    }

    std::vector<struct MetaToken*> token_set;

    // added_tokens is a JSON list of JSON objects
    for (nlohmann::json object : added_tokens) {
        // technically, we can have these in the stack. tbh, not sure if matters.
        struct MetaToken* token = (struct MetaToken*) malloc(sizeof(struct MetaToken));
        token->id               = object["id"];
        token->content          = object["content"];
        token->single_word      = object["single_word"];
        token->lstrip           = object["lstrip"];
        token->rstrip           = object["rstrip"];
        token->normalized       = object["normalized"];
        token->special          = object["special"];
        token_set.push_back(token);
    }

    return token_set;
}

void deallocate_meta_tokens(std::vector<struct MetaToken*> token_set) {
    // Deallocate memory for all struct MetaToken objects
    for (auto token : token_set) {
        delete token;
    }
}

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
    char vocab_name[5] = "BPE"; // NOTE: avoid name conflict

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

    const std::string version = data["version"];
    std::cout << "version: " << version << std::endl;
    const nlohmann::json model_arch = data["truncation"];
    const nlohmann::json padding    = data["padding"];

    // for (const auto &config : data) {
    //     std::cout << "Model Repository:\t" << model_repo << '\n';
    //     std::cout << "Model Architecture:\t" << model_arch << '\n';

    //     // normalizer and pre_tokenizer are either null or an object
    //     const nlohmann::json normalizer    = config["normalizer"];
    //     const nlohmann::json pre_tokenizer = config["pre_tokenizer"];

    //     // NOTE: Normalizer may be one of null, Sequence, NFC, NFD, NFKC, NFKD...
    //     // Seems to be null, Sequence, or NFC in most cases
    //     // Attempt to handle cases where a key may or may not be present and have a null value
    //     // Not sure whether to default to NFD or NFC if normalizer is null
    //     // NOTE: The normalizer type key is not guaranteed
    //     if (!normalizer.is_null()) {
    //         const std::string norm_type = normalizer["type"];
    //         std::cout << "Normalizer Type:\t" << norm_type << std::endl;
    //         // sequence is an array
    //         if (0 == norm_type.compare("Sequence")) {
    //             const nlohmann::json normalizers = normalizer["normalizers"];
    //             if (normalizers.is_array()) {
    //                 for (const auto &norm : normalizers.items()) {
    //                     std::cout << "Norm Sequence Object:\t" << norm.key() << ":\t"
    //                               << norm.value() << std::endl;
    //                 }
    //             }
    //         } else {
    //             // otherwise norm_type is an object potentially containing properties
    //             // this varies from model to model
    //             // maybe we can just dump the entire object? i don't know yet...
    //             for (const auto &norm : normalizer.items()) {
    //                 std::cout << "Pre Object Pair:\t" << norm.key() << ":\t" << norm.value()
    //                           << std::endl;
    //             }
    //         }
    //     }
    // }

    return 0;
}
