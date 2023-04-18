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
