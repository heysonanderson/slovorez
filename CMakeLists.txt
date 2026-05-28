cmake_minimum_required(VERSION 3.15)
project(SlovorezCXX LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.2
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(slovorezCXX)
