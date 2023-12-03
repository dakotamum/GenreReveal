set(distmemcpu_srcs
    main.cpp)
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
include_directories(${MPI_CXX_INCLUDE_PATH})
# might need the next line... was found using 'which mpicc'
# set(CMAKE_C_COMPILER /uufs/chpc.utah.edu/sys/spack/linux-rocky8-nehalem/gcc-8.5.0/intel-mpi-2019.10.317-yy3n4dtopt3qj4qvaeukr6x7xu2suvyk/compilers_and_libraries_2020.4.317/linux/mpi/intel64/bin/mpicc)

add_executable(distmemcpu_exec ${distmemcpu_srcs})
set_target_properties(distmemcpu_exec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/bin OUTPUT_NAME DistMemCpu)
target_link_libraries(distmemcpu_exec PRIVATE shared_lib)
