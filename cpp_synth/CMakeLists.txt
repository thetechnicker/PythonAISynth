cmake_minimum_required(VERSION 3.5)
project(cpp_synth)

set(CMAKE_CXX_STANDARD 14)

# TENSORFLOW C API
include_directories($ENV{TENSORFLOW_API_PATH}/include)
link_directories($ENV{TENSORFLOW_API_PATH}/lib)


# PYTHON dep for numpy
# set(CMAKE_PREFIX_PATH $ENV{CONDA_ENV_PATH})
# find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy REQUIRED)
# include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

add_executable(cpp_synth src/main.cpp)
target_link_libraries(cpp_synth tensorflow)
target_link_libraries(cpp_synth rtmidi)
# target_link_libraries(cpp_synth Python3::Python)


if(MSYS)
    add_custom_command(TARGET cpp_synth POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       $ENV{TENSORFLOW_API_PATH}/lib/tensorflow.dll
                       $<TARGET_FILE_DIR:cpp_synth>)
endif()
