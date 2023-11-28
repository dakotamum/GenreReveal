set(distmemcpu_srcs
    main.cpp)

add_executable(distmemcpu_exec ${distmemcpu_srcs})
set_target_properties(distmemcpu_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/bin OUTPUT_NAME DistMemCpu)
target_link_libraries(distmemcpu_exec PRIVATE csv gcov)
