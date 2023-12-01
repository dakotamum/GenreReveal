set(serial_srcs
    main.cpp)

add_executable(serial_exec ${serial_srcs})
set_target_properties(serial_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/bin OUTPUT_NAME Serial)
target_link_libraries(serial_exec PRIVATE csv gcov)
