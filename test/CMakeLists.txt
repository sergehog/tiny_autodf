cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

add_executable(autodf_test autodf_test.cpp)

target_link_libraries(autodf_test gtest gtest_main pthread)

add_test(NAME autodf_test COMMAND autodf_test)

#cmake_dependent_option(CODE_COVERAGE "Check code-coverage for test targets" ON DCMAKE_BUILD_TYPE EQUALS "Debug")

option(CODE_COVERAGE "Check code-coverage for test targets" OFF)

if(CODE_COVERAGE)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    include(CodeCoverage)
    append_coverage_compiler_flags()

    setup_target_for_coverage_gcovr_html(
            NAME autodf_test_coverage
            EXECUTABLE ctest --verbose
            DEPENDENCIES autodf_test
            BASE_DIRECTORY "../"
            EXCLUDE "test"
    )


endif (CODE_COVERAGE)

