set(openmp_srcs
    main.cpp)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_executable(openmp_exec ${openmp_srcs})
# Link the csv-parser library with our executable
set_target_properties(distmemcpu_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/bin OUTPUT_NAME DistMemCpu)
target_link_libraries(openmp_exec PRIVATE csv gcov)
if(OpenMP_CXX_FOUND)
    target_link_libraries(openmp_exec PUBLIC OpenMP::OpenMP_CXX)
endif()

