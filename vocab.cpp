#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

int main(void) {
    puts("Hello, world!");
    std::ifstream  f("registry.json");
    nlohmann::json data = nlohmann::json::parse(f);
}
