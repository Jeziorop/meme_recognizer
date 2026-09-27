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

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(BZip2 QUIET)
find_package(PNG QUIET)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(EXPAT QUIET)
find_package(X11 QUIET)

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
target_link_libraries(meme_core PUBLIC ${OpenCV_LIBS} Qt6::Core Qt6::Gui freetype EXPAT::EXPAT BZip2::BZip2 PNG::PNG)
target_compile_options(meme_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O3>
)

if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(meme_core PUBLIC CUDA::cudart)
endif()

if(TARGET Fontconfig::Fontconfig)
    set_property(TARGET Fontconfig::Fontconfig APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES freetype EXPAT::EXPAT BZip2::BZip2 PNG::PNG
    )
endif()

# Main Desktop GUI Application
add_executable(meme_recognizer
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.hpp
)
target_link_libraries(meme_recognizer PRIVATE meme_core Qt6::Widgets)
if(TARGET Qt6::QOffscreenIntegrationPlugin)
    target_link_libraries(meme_recognizer PRIVATE Qt6::QOffscreenIntegrationPlugin)
    target_compile_definitions(meme_recognizer PRIVATE MEME_HAVE_STATIC_OFFSCREEN_PLUGIN=1)
endif()
if(X11_FOUND)
    target_compile_definitions(meme_recognizer PRIVATE MEME_HAVE_X11=1)
    target_include_directories(meme_recognizer PRIVATE ${X11_INCLUDE_DIR})
    target_link_libraries(meme_recognizer PRIVATE ${X11_LIBRARIES})
endif()
target_link_libraries(meme_recognizer PRIVATE ${STATIC_FONT_GROUP})
target_compile_options(meme_recognizer PRIVATE -Wall -Wextra -Wpedantic -O3)

# Automated Unit & Integration Test Suite
enable_testing()
add_executable(meme_recognizer_tests
    tests/test_meme_recognizer.cpp
)
target_link_libraries(meme_recognizer_tests PRIVATE meme_core Qt6::Widgets)
if(TARGET Qt6::QOffscreenIntegrationPlugin)
    target_link_libraries(meme_recognizer_tests PRIVATE Qt6::QOffscreenIntegrationPlugin)
endif()
target_link_libraries(meme_recognizer_tests PRIVATE ${STATIC_FONT_GROUP})
add_test(NAME MemeRecognizerEndToEndTests COMMAND meme_recognizer_tests
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
