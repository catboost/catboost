
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.


if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND UNIX AND NOT APPLE AND NOT ANDROID)
  include(CMakeLists.linux-aarch64.txt)
elseif (APPLE)
  include(CMakeLists.darwin.txt)
elseif (ANDROID AND CMAKE_ANDROID_ARCH STREQUAL "arm")
  include(CMakeLists.android-arm.txt)
elseif (ANDROID AND CMAKE_ANDROID_ARCH STREQUAL "arm64")
  include(CMakeLists.android-arm64.txt)
elseif (ANDROID AND CMAKE_ANDROID_ARCH STREQUAL "x86")
  include(CMakeLists.android-x86.txt)
elseif (ANDROID AND CMAKE_ANDROID_ARCH STREQUAL "x86_64")
  include(CMakeLists.android-x86_64.txt)
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND UNIX AND NOT APPLE AND NOT ANDROID)
  include(CMakeLists.linux.txt)
endif()
