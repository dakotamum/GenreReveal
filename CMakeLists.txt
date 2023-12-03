# followed this tutorial to add the csv parser as dependency: https://medium.com/@justanotherdev/parsing-csvs-with-modern-c-b8c3cac3c742
cmake_minimum_required(VERSION 3.9)
project(GenreReveal)
set(CMAKE_CXX_STANDARD 17)

# set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/bin)

# clone csv-parser
set(csv-parser_dir ${CMAKE_SOURCE_DIR}/csv-parser)
if(NOT EXISTS ${csv-parser_dir})
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/vincentlaucsb/csv-parser.git ${csv-parser_dir}
    )
endif()
# link csv-parser library with the project
add_subdirectory(csv-parser)
include_directories(${csv-parser_dir}/single_include)

# include each subdirectory
add_subdirectory(Serial)
add_subdirectory(DistMemCpu)
add_subdirectory(OpenMp)
add_subdirectory(SharedMemGpu)

# copy python script to build directory
configure_file(${CMAKE_SOURCE_DIR}/plot.py ${CMAKE_BINARY_DIR}/dist/bin/plot.py COPYONLY)
