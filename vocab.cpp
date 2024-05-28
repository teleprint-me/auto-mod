#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char* argv[]) {
    const char* const   short_options = "f:";
    const struct option long_options[]
        = {{"registry-file", required_argument, nullptr, 'f'}, {nullptr}};

    int  opt;
    char registryFilePath[1024];

    while ((opt = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                strcpy(registryFilePath, optarg);
                break;

            default:
                puts("Usage: vocab [-f <file>]");
                return 1;
        }
    }

    std::ifstream  f(registryFilePath);
    nlohmann::json data = nlohmann::json::parse(f);

    if (data.is_null()) {
        puts("Error: Unable to parse registry file.");
        return 1;
    }

    for (const auto &config : data) {
        const std::string modelArch = config["model_arch"];

        if (modelArch == "stablelm") {
            std::cout << "Model Architecture:\t" << modelArch << '\n';

            // normalizer and pre_tokenizer are either null or an object
            const nlohmann::json normalizer    = config["normalizer"];
            const nlohmann::json pre_tokenizer = config["pre_tokenizer"];

            // Handle the cases where a key may or may not be present and have a null value
            if (!normalizer.is_null()) {
                // NOTE: The normalizer type key is not guaranteed
                const std::string norm_type = normalizer["type"];
                // i don't know how to check for a "truthy" string in C++
                std::cout << "Normalizer Type:" << norm_type << std::endl;
                // sequence is an array
                if (0 == norm_type.compare("Sequence")) {
                    // extract the array
                    // each element in the array is an object
                } else {
                    // otherwise norm_type is an object potentially containing properties
                    // this varies from model to model
                    // maybe we can just dump the entire object?
                    // i don't know yet
                }
            }

            if (!pre_tokenizer.is_null()) {
                // NOTE: The pre_tokenizer type is not guaranteed
                const std::string pre_tokenizer_type = pre_tokenizer["type"];
                // i don't know how to check for a "truthy" string in C++
                std::cout << "Pre-tokenizer Type:" << pre_tokenizer_type << std::endl;
                // sequence is an array
                if (0 == pre_tokenizer_type.compare("Sequence")) {
                    // extract the array
                    // each element in the array is an object
                } else {
                    // otherwise pre_tokenizer_type is an object potentially containing properties
                    // this varies from model to model
                    // maybe we can just dump the entire object?
                    // i don't know yet
                }
            }
        }
    }

    return 0;
}
