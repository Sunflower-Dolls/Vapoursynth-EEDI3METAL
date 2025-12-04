cmake_minimum_required(VERSION 3.20)
project(eedi3metalDylib LANGUAGES OBJCXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(VAPOURSYNTH REQUIRED vapoursynth)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(eedi3metal SHARED
    eedi3metal.mm
)

target_include_directories(eedi3metal PRIVATE 
    ${VAPOURSYNTH_INCLUDE_DIRS}
)

option(DUMP_METAL_BUFFERS "Enable dumping of intermediate Metal buffers" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers for detecting possible errors" OFF)
option(ENABLE_PROFILING "Enable profiling of the kernel" OFF)

if(DUMP_METAL_BUFFERS)
    target_compile_definitions(eedi3metal PRIVATE DUMP_METAL_BUFFERS)
endif()

if(ENABLE_SANITIZERS)
    target_compile_options(eedi3metal PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(eedi3metal PRIVATE -fsanitize=address,undefined)
endif()

if(ENABLE_PROFILING)
    target_compile_definitions(eedi3metal PRIVATE DEBUG_PROFILING)
endif()

set_target_properties(eedi3metal PROPERTIES 
    CXX_FLAGS "-Wno-c23-extensions -mmacosx-version-min=12.0"
    COMPILE_FLAGS "-fobjc-arc"
)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

target_link_libraries(eedi3metal PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)
