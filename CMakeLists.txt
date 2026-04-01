cmake_minimum_required(VERSION 3.10)
project(MetalNet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR})

# Find all source files
file(GLOB_RECURSE SOURCES 
    "main.cpp"
    "core/*.cpp"
    "Layers/*.cpp"
    "nn/*.cpp"
)

# Create the executable
add_executable(MetalNet ${SOURCES})
