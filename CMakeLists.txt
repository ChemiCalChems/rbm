cmake_minimum_required(VERSION 3.20)
project(rbm)

set(CMAKE_CXX_STANDARD 23)

find_package(TBB)
find_package(OpenMP REQUIRED)

add_executable(train train.cpp)
add_executable(reconstruct reconstruct.cpp)

target_link_libraries(train TBB::tbb OpenMP::OpenMP_CXX)
target_link_libraries(reconstruct TBB::tbb OpenMP::OpenMP_CXX)
