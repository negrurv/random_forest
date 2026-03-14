# backend/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(RandomForestProject)

# C++17 is required for some of the modern features we used
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# -O3 is critical: it tells the compiler to optimize the math for maximum speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

include_directories(include)

# Download pybind11 directly from GitHub
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Create the Python module named "rf_cpp"
pybind11_add_module(rf_cpp 
    src/bindings.cpp 
    src/random_forest.cpp
)