set(openmp_srcs
    main.cpp)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


add_executable(openmp_exec ${openmp_srcs})
# Link the csv-parser library with our executable
set_target_properties(openmp_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/bin OUTPUT_NAME openmp_exec)
target_link_libraries(openmp_exec PRIVATE shared_lib)
target_link_libraries(openmp_exec PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(openmp_exec PRIVATE ${OpenMP_CXX_FLAGS})

