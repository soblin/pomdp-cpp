project(pomdp-cpp)
cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -g -std=c++2a")

find_package(xtensor)

set(${PROJECT_NAME}_INCLUDE_DIRS
  include/
)

add_executable(tiger src/tiger.cpp)
target_include_directories(tiger PUBLIC ${xtensor_INCLUDE_DIRS} ${${PROJECT_NAME}_INCLUDE_DIRS})

add_executable(lightdark src/lightdark.cpp)
target_include_directories(lightdark PUBLIC ${xtensor_INCLUDE_DIRS} ${${PROJECT_NAME}_INCLUDE_DIRS})

# add valgrind test: valgrind --tool=memcheck --leak-check=yes --undef-value-errors=no <executable>
