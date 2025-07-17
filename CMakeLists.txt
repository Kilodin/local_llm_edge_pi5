cmake_minimum_required(VERSION 3.16)
project(local_llm_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Add llama.cpp as subdirectory
add_subdirectory(third_party/llama.cpp)

# Include directories
include_directories(
    src/cpp
    third_party/llama.cpp
)

# Core library
add_library(llm_core STATIC
    src/cpp/model/llm_model.cpp
    src/cpp/inference/inference_engine.cpp
    src/cpp/inference/prompt_processor.cpp
)

target_link_libraries(llm_core
    llama
    Threads::Threads
)

# Node.js binding
add_library(llm_node SHARED
    src/cpp/bindings/node_binding.cpp
)

target_link_libraries(llm_node
    llm_core
    node-addon-api
)

# Set output directory
set_target_properties(llm_node PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
) 