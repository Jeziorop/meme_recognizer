cmake_minimum_required(VERSION 3.25)

# Prefer /usr/bin/nvcc (CUDA 12.0) which supports Pascal sm_61 (Quadro P620)
if(EXISTS "/usr/bin/nvcc" AND NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "61" CACHE STRING "Target CUDA Architecture (sm_61 for Quadro P620)")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(meme_recognizer VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(MEME_ENABLE_CUDA=1)
    message(STATUS "CUDA GPU acceleration enabled with compiler: ${CMAKE_CUDA_COMPILER} (arch=${CMAKE_CUDA_ARCHITECTURES})")
else()
    add_compile_definitions(MEME_ENABLE_CUDA=0)
    message(STATUS "CUDA compiler not found; falling back to OpenCV DNN OpenCL/CPU backend")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)

set(CORE_SOURCES
    src/GpuDeviceManager.cpp
    src/PoseFeatureExtractor.cpp
    src/PoseDetector.cpp
    src/MemeClassifier.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND CORE_SOURCES src/CudaInference.cu)
endif()

add_library(meme_core STATIC ${CORE_SOURCES})
target_include_directories(meme_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(meme_core PUBLIC ${OpenCV_LIBS})
target_compile_options(meme_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O3>
)

if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(meme_core PUBLIC CUDA::cudart)
endif()

# Dear ImGui + OpenGL Backend Static Library
add_library(imgui_lib STATIC
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/backends/imgui_impl_opengl2.cpp
    third_party/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends
)
target_link_libraries(imgui_lib PUBLIC OpenGL::GL ${CMAKE_DL_LIBS})
target_compile_options(imgui_lib PRIVATE -O3)

# Main Desktop GUI Application (Dear ImGui + OpenGL + X11/GLX)
add_executable(meme_recognizer
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.hpp
)
target_include_directories(meme_recognizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends
    ${X11_INCLUDE_DIR}
)
target_link_libraries(meme_recognizer PRIVATE
    meme_core
    imgui_lib
    OpenGL::GL
    ${X11_LIBRARIES}
    ${CMAKE_DL_LIBS}
)
target_compile_options(meme_recognizer PRIVATE -Wall -Wextra -Wpedantic -O3)

# Automated Unit & Integration Test Suite
enable_testing()
add_executable(meme_recognizer_tests
    tests/test_meme_recognizer.cpp
)
target_link_libraries(meme_recognizer_tests PRIVATE meme_core)
add_test(NAME MemeRecognizerEndToEndTests COMMAND meme_recognizer_tests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
