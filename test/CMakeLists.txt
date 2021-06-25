cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 14)

find_package(GTest CONFIG REQUIRED)

add_executable(autodf_test autodf_test.cpp)

target_link_libraries(autodf_test gtest gtest_main pthread)

add_test(NAME pga_test COMMAND pga_test)

