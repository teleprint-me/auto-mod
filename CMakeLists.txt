cmake_minimum_required(VERSION 3.17)
project("auto-mod" LANGUAGES CXX C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(vocab src/vocab.cpp)
add_executable(magic src/magic.cpp)
