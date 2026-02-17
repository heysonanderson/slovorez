cmake_minimum_required(VERSION 3.15)
project(SlovorezCXX LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lexer lexerrun.cc text_lexer.cc utf8_decoder.h)
install(TARGETS lexer RUNTIME)
