#libFL CMAKE

cmake_minimum_required(VERSION 2.8.12)

project(libFL C CXX)

set(CMAKE_VERBOSE_MAKEFILE "ON")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -fopenmp=libomp")

find_package(Threads REQUIRED)

message(STATUS ${CMAKE_MODULE_PATH})

find_package( CBLAS REQUIRED )

find_package( LAPACK REQUIRED )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

option (USE_LAPACKE "Use LAPACKE C interface to LAPACK" ON)
option (USE_LAPACKE "Use BLAS C interface to BLAS" ON)
option (USE_OPENMP "Use OpenMP for multithreading" ON)
 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(include src .)

message(Compiler: "${CMAKE_CXX_COMPILER_ID}")

message(Compiler: "${CMAKE_C_COMPILER_ID}")

message(Source dir: "${CMAKE_SOURCE_DIR}")
 
file(GLOB LIBFL_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB LIBFL_INCLUDE ${CMAKE_SOURCE_DIR}/include/*.h)

message(STATUS "LIBFL library")
foreach( libflsourcefile ${LIBFL_SRC} )
    message(STATUS "\t${libflsourcefile}")
endforeach(libflsourcefile ${LIBFL_SRC})

file(GLOB EXTERNAL_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/externals/*/include")
file(GLOB EXTERNAL_LIBRARIES "${CMAKE_SOURCE_DIR}/externals/*/lib/*.a")
    
add_library(fl STATIC ${LIBFL_SRC} ${LIBFL_INCLUDE})

target_include_directories(fl PUBLIC ${EXTERNAL_INCLUDE_DIRS})
target_link_libraries(fl
                      ${CMAKE_SOURCE_DIR}/externals/libjpeg/lib/libjpeg.a
                      ${CMAKE_SOURCE_DIR}/externals/libpng/lib/libpng.a
                      ${CMAKE_SOURCE_DIR}/externals/libsvm/lib/libsvm.a
                      ${CMAKE_SOURCE_DIR}/externals/isf/lib/libift.a
                      ${CMAKE_SOURCE_DIR}/externals/zlib/lib/zlib.a)

file(GLOB LIBFL_DEMO
    ${CMAKE_SOURCE_DIR}/demo/*.cpp
    )

install(FILES ${LIBFL_INCLUDE}
        DESTINATION include
        )

message(STATUS "Demo files")
foreach( demosourcefile ${LIBFL_DEMO} )
    get_filename_component(demofilename ${demosourcefile} NAME)
    string( REPLACE ".cpp" "" demoname ${demofilename} )

    message(STATUS "\t${demosourcefile}")

    add_executable( ${demoname} ${demosourcefile})
    target_link_libraries( ${demoname} fl ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES} stdc++)

endforeach( demosourcefile ${LIBFL_DEMO} )

