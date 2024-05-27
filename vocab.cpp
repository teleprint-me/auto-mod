#include <cstring>
#include <fstream>
#include <getopt.h>
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

    puts("Hello, world!");
    std::ifstream  f(registryFilePath);
    nlohmann::json data = nlohmann::json::parse(f);
}
