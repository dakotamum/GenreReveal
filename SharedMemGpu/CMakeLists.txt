# followed this tutorial to add the csv parser as dependency: https://medium.com/@justanotherdev/parsing-csvs-with-modern-c-b8c3cac3c742
cmake_minimum_required(VERSION 3.9)
project(GenreReveal CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

# Clone csv-parser
set(CSVPARSER_DIR ${CMAKE_SOURCE_DIR}/external/lib/csvparser)
if(NOT EXISTS ${CSVPARSER_DIR})
    find_package(Git REQUIRED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/vincentlaucsb/csv-parser.git ${CSVPARSER_DIR}
    )
endif()

# Locate CUDA
find_package(CUDAToolkit REQUIRED)

# Add your CUDA files to the project
set(CUDA_SOURCE_FILES
    cuda_kMeans.cu
    cuda_kMeans.cuh
    # Add more CUDA source files if you have more
)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

# Specify the executable and its source files
add_executable(${PROJECT_NAME} main.cpp ${CUDA_SOURCE_FILES})
# target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_ROOT})

# Specify CUDA properties for the target executable
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11  # Specify the desired CUDA standard version
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 60
)


# Link CUDA libraries and dependencies
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    CUDA::cudart  # Link the CUDA runtime library
    PRIVATE
    shared_lib 
    # Add any other necessary libraries or dependencies
)


configure_file(
    "${CMAKE_SOURCE_DIR}/plot.py"
    "${CMAKE_BINARY_DIR}/plot.py"
    COPYONLY
)
