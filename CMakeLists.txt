cmake_minimum_required (VERSION 3.9)

project(deterministic LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

add_executable(demmel2013
  src/demmel2013.cpp
)
target_compile_features(demmel2013 PRIVATE cxx_std_14)
target_compile_options(demmel2013 PRIVATE -Wall -pedantic)
target_link_libraries(demmel2013 PRIVATE OpenMP::OpenMP_CXX)

add_executable(ahrens2020
  src/ahrens2020.cpp
)
target_compile_features(ahrens2020 PRIVATE cxx_std_17)
target_compile_options(ahrens2020 PRIVATE -Wall -pedantic)

add_executable(ahrens2020_cuda
  src/ahrens2020_cuda.cu
)
target_compile_features(ahrens2020_cuda PRIVATE cxx_std_17)
target_compile_options(ahrens2020_cuda PRIVATE -Wall -Wextra --expt-relaxed-constexpr -Xptxas="-v" -lineinfo)
