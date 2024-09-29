cmake_minimum_required(VERSION 3.15)

project(VkFluidSimulation VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBLAVA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/liblava)

set(LIBLAVA_DEMO OFF)
set(LIBLAVA_TEST OFF)
set(LIBLAVA_TEMPLATE OFF)

set(SPIRV_WERROR OFF CACHE BOOL "Disable warnings as errors in SPIRV-Tools")
set(SPIRV_WARNINGS_AS_ERRORS OFF CACHE BOOL "Disable warnings as errors in SPIRV-Tools")

add_subdirectory(${LIBLAVA_DIR})

if(MSVC)
    add_compile_options(/WX-)

    add_compile_options(/wd4717)
endif()

add_executable(VkFluidSimulation src/main.cpp)

target_link_libraries(VkFluidSimulation PRIVATE
    lava::engine 
    ${LIBLAVA_ENGINE_LIBRARIES}
)

if (TARGET SPIRV-Tools-opt)
    target_compile_options(SPIRV-Tools-opt PRIVATE /WX- /wd4717)
endif()

if (TARGET SPIRV-Tools-static)
    target_compile_options(SPIRV-Tools-static PRIVATE /WX- /wd4717)
endif()
