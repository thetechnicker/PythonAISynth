cmake_minimum_required(VERSION 3.5)
project(cpp_synth)

set(CMAKE_CXX_STANDARD 14)

# TENSORFLOW C API
set(TENSORFLOW_API_PATH $ENV{TENSORFLOW_API_PATH})
include_directories(${TENSORFLOW_API_PATH}/include)
link_directories(${TENSORFLOW_API_PATH}/lib)

add_executable(cpp_synth src/main.cpp)
target_link_libraries(cpp_synth tensorflow)

if(MSYS)
    add_custom_command(TARGET cpp_synth POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${TENSORFLOW_API_PATH}/lib/tensorflow.dll
                       $<TARGET_FILE_DIR:cpp_synth>)
endif()
