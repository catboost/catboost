cmake_minimum_required(VERSION 2.6)

# The matrixnet library project.
project(catboost_calc_example)

add_definitions(-std=c++11)

IF (WIN32)
add_library(catboostmodel STATIC IMPORTED)
set_property(TARGET catboostmodel PROPERTY IMPORTED_LOCATION "catboostmodel.lib")
ELSE()
IF(APPLE)
add_library(catboostmodel SHARED IMPORTED)
set_property(TARGET catboostmodel PROPERTY IMPORTED_LOCATION "libcatboostmodel.dylib")
ELSE()
add_library(catboostmodel SHARED IMPORTED)
set_property(TARGET catboostmodel PROPERTY IMPORTED_LOCATION "libcatboostmodel.so")
ENDIF()
ENDIF()

add_executable(catboost_demo main.cpp)

target_link_libraries(catboost_demo catboostmodel)
