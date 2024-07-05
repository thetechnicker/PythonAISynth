cmake_minimum_required(VERSION 3.5)
project(cpp_synth)

set(CMAKE_CXX_STANDARD 14)


# TENSORFLOW C API
include_directories($ENV{TENSORFLOW_API_PATH}/include)
include_directories(include/)
link_directories($ENV{TENSORFLOW_API_PATH}/lib)

file(GLOB SOURCES src/*.c*)
add_executable(cpp_synth ${SOURCES})

target_link_libraries(cpp_synth tensorflow)
target_link_libraries(cpp_synth rtmidi)

find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
target_link_libraries(cpp_synth sfml-graphics sfml-audio)


if(MSYS)
    add_custom_command(TARGET cpp_synth POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       $ENV{TENSORFLOW_API_PATH}/lib/tensorflow.dll
                       $<TARGET_FILE_DIR:cpp_synth>)
endif()
