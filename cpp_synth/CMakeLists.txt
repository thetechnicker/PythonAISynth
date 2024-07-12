cmake_minimum_required(VERSION 3.5)
project(cpp_synth)

set(CMAKE_C_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH "~/.local")

include_directories(include/)
file(GLOB SOURCES src/*.c*)
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_executable(cpp_synth ${SOURCES})


target_link_libraries(cpp_synth rtmidi)

if(MSYS OR WIN32)
    include_directories($ENV{TENSORFLOW_API_PATH}/include)
    link_directories($ENV{TENSORFLOW_API_PATH}/lib)
    target_link_libraries(cpp_synth tensorflow)

    set(SFML_DIR $ENV{SFML_API_PATH}/SFML-2.6.1/lib/cmake/SFML)
   
    add_custom_command(TARGET cpp_synth POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       $ENV{TENSORFLOW_API_PATH}/lib/tensorflow.dll
                       $<TARGET_FILE_DIR:cpp_synth>)

    add_custom_command(TARGET cpp_synth POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       $ENV{SFML_API_PATH}/SFML-2.6.1/bin/
                       $<TARGET_FILE_DIR:cpp_synth>)
elseif(EXISTS "/mnt/c/Windows/System32" OR UNIX)
    #using WSL
    find_package(TensorFlow CONFIG REQUIRED)
    target_link_libraries(cpp_synth TensorFlow::TensorFlow)

endif()

find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
target_link_libraries(cpp_synth sfml-graphics sfml-audio)
