cmake_minimum_required(VERSION 3.10)
project(MetalNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR})

# Find all source files
file(GLOB_RECURSE SOURCES 
    "main.cpp"
    "core/*.cpp"
    "Layers/*.cpp"
    "nn/*.cpp"
    "data/*.cpp"
)

# Find OpenMP
find_package(OpenMP)

# Create the executable
add_executable(MetalNet ${SOURCES})

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(MetalNet PRIVATE OpenMP::OpenMP_CXX)
endif()
