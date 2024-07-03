cmake_minimum_required(VERSION 3.0)
project(cpp_synth)

# TENSORFLOW C API
set(TENSORFLOW_API_PATH $ENV{TENSORFLOW_API_PATH})
include_directories(${TENSORFLOW_API_PATH}/include)
link_directories(${TENSORFLOW_API_PATH}/lib)

add_executable(cpp_synth src/main.cpp)
target_link_libraries(cpp_synth tensorflow)