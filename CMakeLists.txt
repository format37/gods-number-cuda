cmake_minimum_required(VERSION 3.18)
project(rubic_gods_number CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Target RTX 4090 (Ada Lovelace, compute capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(gods_number
    src/main.cu
)

target_include_directories(gods_number PRIVATE src)

target_compile_options(gods_number PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)
