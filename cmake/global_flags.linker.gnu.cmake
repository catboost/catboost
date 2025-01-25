if (ANDROID)
  # NDK r23 onwards has stopped using libgcc:
  # - https://github.com/android/ndk/wiki/Changelog-r23#changes
  # - https://github.com/android/ndk/issues/1230
  #   LLVM's libunwind is now used instead of libgcc for all architectures rather than just 32-bit Arm.
  # - https://github.com/android/ndk/issues/1231
  #   LLVM's libclang_rt.builtins is now used instead of libgcc.
  if (CMAKE_ANDROID_NDK_VERSION GREATER_EQUAL 23)
    # Use toolchain defaults to link with libunwind/clang_rt.builtins
    add_link_options("-nostdlib++")
  else ()
    # Preserve old behaviour: specify runtime libs manually
    add_link_options(-nodefaultlibs)
    link_libraries(gcc)
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      link_libraries(unwind)
    endif()
  endif()
elseif (CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux)$")
  add_link_options("-nodefaultlibs")
endif()

if (APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^(Android|Linux)$")
  add_link_options(-rdynamic)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # NB:
  #   glibc merged pthread into libc.so as of 2.34 / Ubuntu 22.04, see
  #   https://developers.redhat.com/articles/2021/12/17/why-glibc-234-removed-libpthread
  #
  # In macOS and iOS libpthread points to libSystem already (just as libc.tbd does):
  #   $ file libpthread.tbd
  #   libpthread.tbd: symbolic link to libSystem.tbd
  #
  # Android does not provide libpthread at all.
  # Once we will be building against glibc=2.34, we might simply remove -lpthread
  link_libraries(pthread)
endif()
