cmake_minimum_required(VERSION 3.17)
project(tokenizers LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(vocab vocab.cpp)
