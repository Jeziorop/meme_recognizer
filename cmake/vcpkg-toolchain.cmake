# Portable toolchain wrapper:
# 1. If global VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake is accessible and vcpkg_installed is not yet populated,
#    delegates to vcpkg.cmake to run `vcpkg install`.
# 2. If build/default/vcpkg_installed/x64-linux is already populated (or VCPKG_ROOT is outside an isolated build sandbox),
#    configures CMAKE_PREFIX_PATH directly from ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-linux.

set(_LOCAL_VCPKG_INSTALLED "${CMAKE_CURRENT_LIST_DIR}/../build/default/vcpkg_installed/x64-linux")

if(EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" AND NOT EXISTS "${_LOCAL_VCPKG_INSTALLED}/share/opencv4/OpenCVConfig.cmake")
    include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
elseif(EXISTS "${_LOCAL_VCPKG_INSTALLED}")
    list(PREPEND CMAKE_PREFIX_PATH "${_LOCAL_VCPKG_INSTALLED}")
    list(PREPEND CMAKE_LIBRARY_PATH "${_LOCAL_VCPKG_INSTALLED}/lib" "${_LOCAL_VCPKG_INSTALLED}/lib/manual-link")
    list(PREPEND CMAKE_INCLUDE_PATH "${_LOCAL_VCPKG_INSTALLED}/include")
    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
elseif(EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    include("$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
