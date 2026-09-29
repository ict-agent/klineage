cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(GemmBF16Opt LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
)

set_source_files_properties(op_kernel/our_matmul_entry.cpp PROPERTIES LANGUAGE ASC)
set_source_files_properties(host/our_main.cpp PROPERTIES LANGUAGE ASC)

add_executable(our_matmul host/our_main.cpp op_kernel/our_matmul_entry.cpp)
target_include_directories(our_matmul PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    ${ASCEND_HOME_PATH}/include/aclnn)
link_directories(${ASCEND_HOME_PATH}/lib64)
target_link_libraries(our_matmul PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager ascendcl tiling_api
    platform register graph m)
