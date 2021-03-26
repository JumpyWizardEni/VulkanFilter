cmake_minimum_required (VERSION 3.8)
project (vulkan_minimal_compute)

find_package(Vulkan)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-g -fopenmp")
include_directories(${Vulkan_INCLUDE_DIR})
set(ALL_LIBS  ${Vulkan_LIBRARY} )

add_executable(vulkan_minimal_compute src/main.cpp src/vk_utils.h src/vk_utils.cpp src/Bitmap.h src/Bitmap.cpp src/bilateral.cpp)

set_target_properties(vulkan_minimal_compute PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(vulkan_minimal_compute ${ALL_LIBS} )

file(COPY shaders/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)
