project(shared_lib)

set(lib_srcs
    Config.hpp
    Point.hpp
    readcsv.hpp
    serialVerify.hpp
    timer.hpp
)

add_library(shared_lib ${lib_srcs})
target_link_libraries(shared_lib PRIVATE csv gcov)
target_include_directories(shared_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
