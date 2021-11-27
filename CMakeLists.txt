cmake_minimum_required(VERSION 3.15)
project(tiny_autodf)

set(CMAKE_CXX_STANDARD 14)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE .)

add_subdirectory(test)
