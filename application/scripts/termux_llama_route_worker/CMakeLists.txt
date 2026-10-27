# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
cmake_minimum_required(VERSION 3.18)
project(termux_llama_route_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR is required. Example: -DLLAMA_CPP_DIR=$ENV{HOME}/llama.cpp")
endif()

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR does not point to a llama.cpp source tree: ${LLAMA_CPP_DIR}")
endif()

add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp-build" EXCLUDE_FROM_ALL)

add_executable(termux_llama_route_worker ../termux_llama_route_worker.cpp)
target_link_libraries(termux_llama_route_worker PRIVATE llama)
