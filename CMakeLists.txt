cmake_minimum_required(VERSION 3.0)
project(speedup_aarch64_cpu)

option(USE_NEON_A55 "set use neon target on A55" ON)
option(USE_NEON_A76 "set use neon target on A76" OFF)

include_directories(include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O3 -march=armv8.2-a")

if (USE_NEON_A55)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_NEON_A55")
elseif (USE_NEON_A76)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_NEON_A76")
endif(USE_NEON_A55)

add_subdirectory(src)
add_subdirectory(test)