# The MIT License (MIT)
#
# Copyright (c) 2024 JFrog
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(CONAN_MINIMUM_VERSION 2.0.5)

# Create a new policy scope and set the minimum required cmake version so the
# features behind a policy setting like if(... IN_LIST ...) behaves as expected
# even if the parent project does not specify a minimum cmake version or a minimum
# version less than this module requires (e.g. 3.0) before the first project() call.
# (see: https://cmake.org/cmake/help/latest/variable/CMAKE_PROJECT_TOP_LEVEL_INCLUDES.html)
#
# The policy-affecting calls like cmake_policy(SET...) or `cmake_minimum_required` only
# affects the current policy scope, i.e. between the PUSH and POP in this case.
#
# https://cmake.org/cmake/help/book/mastering-cmake/chapter/Policies.html#the-policy-stack
cmake_policy(PUSH)
cmake_minimum_required(VERSION 3.24)

function(detect_os OS OS_API_LEVEL OS_SDK OS_SUBSYSTEM OS_VERSION)
    # it could be cross compilation
    message(STATUS "CMake-Conan: cmake_system_name=${CMAKE_SYSTEM_NAME}")
    if(CMAKE_SYSTEM_NAME AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")
        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            set(${OS} Macos PARENT_SCOPE)
        elseif(CMAKE_SYSTEM_NAME STREQUAL "QNX")
            set(${OS} Neutrino PARENT_SCOPE)
        elseif(CMAKE_SYSTEM_NAME STREQUAL "CYGWIN")
            set(${OS} Windows PARENT_SCOPE)
            set(${OS_SUBSYSTEM} cygwin PARENT_SCOPE)
        elseif(CMAKE_SYSTEM_NAME MATCHES "^MSYS")
            set(${OS} Windows PARENT_SCOPE)
            set(${OS_SUBSYSTEM} msys2 PARENT_SCOPE)
        else()
            set(${OS} ${CMAKE_SYSTEM_NAME} PARENT_SCOPE)
        endif()
        if(CMAKE_SYSTEM_NAME STREQUAL "Android")
            if(DEFINED ANDROID_PLATFORM)
                string(REGEX MATCH "[0-9]+" _OS_API_LEVEL ${ANDROID_PLATFORM})
            elseif(DEFINED CMAKE_SYSTEM_VERSION)
                set(_OS_API_LEVEL ${CMAKE_SYSTEM_VERSION})
            endif()
            message(STATUS "CMake-Conan: android api level=${_OS_API_LEVEL}")
            set(${OS_API_LEVEL} ${_OS_API_LEVEL} PARENT_SCOPE)
        endif()
        if(CMAKE_SYSTEM_NAME MATCHES "Darwin|iOS|tvOS|watchOS")
            # CMAKE_OSX_SYSROOT contains the full path to the SDK for MakeFile/Ninja
            # generators, but just has the original input string for Xcode.
            if(NOT IS_DIRECTORY ${CMAKE_OSX_SYSROOT})
                set(_OS_SDK ${CMAKE_OSX_SYSROOT})
            else()
                if(CMAKE_OSX_SYSROOT MATCHES Simulator)
                    set(apple_platform_suffix simulator)
                else()
                    set(apple_platform_suffix os)
                endif()
                if(CMAKE_OSX_SYSROOT MATCHES AppleTV)
                    set(_OS_SDK "appletv${apple_platform_suffix}")
                elseif(CMAKE_OSX_SYSROOT MATCHES iPhone)
                    set(_OS_SDK "iphone${apple_platform_suffix}")
                elseif(CMAKE_OSX_SYSROOT MATCHES Watch)
                    set(_OS_SDK "watch${apple_platform_suffix}")
                endif()
            endif()
            if(DEFINED _OS_SDK)
                message(STATUS "CMake-Conan: cmake_osx_sysroot=${CMAKE_OSX_SYSROOT}")
                set(${OS_SDK} ${_OS_SDK} PARENT_SCOPE)
            endif()
            if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
                message(STATUS "CMake-Conan: cmake_osx_deployment_target=${CMAKE_OSX_DEPLOYMENT_TARGET}")
                set(${OS_VERSION} ${CMAKE_OSX_DEPLOYMENT_TARGET} PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()


function(detect_arch ARCH)
    # CMAKE_OSX_ARCHITECTURES can contain multiple architectures, but Conan only supports one.
    # Therefore this code only finds one. If the recipes support multiple architectures, the
    # build will work. Otherwise, there will be a linker error for the missing architecture(s).
    if(DEFINED CMAKE_OSX_ARCHITECTURES)
        string(REPLACE " " ";" apple_arch_list "${CMAKE_OSX_ARCHITECTURES}")
        list(LENGTH apple_arch_list apple_arch_count)
        if(apple_arch_count GREATER 1)
            message(WARNING "CMake-Conan: Multiple architectures detected, this will only work if Conan recipe(s) produce fat binaries.")
        endif()
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin|iOS|tvOS|watchOS" AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(host_arch ${CMAKE_OSX_ARCHITECTURES})
    elseif(MSVC)
        set(host_arch ${CMAKE_CXX_COMPILER_ARCHITECTURE_ID})
    else()
        set(host_arch ${CMAKE_SYSTEM_PROCESSOR})
    endif()
    if(host_arch MATCHES "aarch64|arm64|ARM64")
        set(_ARCH armv8)
    elseif(host_arch MATCHES "armv7|armv7-a|armv7l|ARMV7")
        set(_ARCH armv7)
    elseif(host_arch MATCHES armv7s)
        set(_ARCH armv7s)
    elseif(host_arch MATCHES "i686|i386|X86")
        set(_ARCH x86)
    elseif(host_arch MATCHES "AMD64|amd64|x86_64|x64")
        set(_ARCH x86_64)
    endif()
    message(STATUS "CMake-Conan: cmake_system_processor=${_ARCH}")
    set(${ARCH} ${_ARCH} PARENT_SCOPE)
endfunction()


function(detect_cxx_standard CXX_STANDARD)
    set(${CXX_STANDARD} ${CMAKE_CXX_STANDARD} PARENT_SCOPE)
    if(CMAKE_CXX_EXTENSIONS)
        set(${CXX_STANDARD} "gnu${CMAKE_CXX_STANDARD}" PARENT_SCOPE)
    endif()
endfunction()


macro(detect_gnu_libstdcxx)
    # _CONAN_IS_GNU_LIBSTDCXX true if GNU libstdc++
    check_cxx_source_compiles("
    #include <cstddef>
    #if !defined(__GLIBCXX__) && !defined(__GLIBCPP__)
    static_assert(false);
    #endif
    int main(){}" _CONAN_IS_GNU_LIBSTDCXX)

    # _CONAN_GNU_LIBSTDCXX_IS_CXX11_ABI true if C++11 ABI
    check_cxx_source_compiles("
    #include <string>
    static_assert(sizeof(std::string) != sizeof(void*), \"using libstdc++\");
    int main () {}" _CONAN_GNU_LIBSTDCXX_IS_CXX11_ABI)

    set(_CONAN_GNU_LIBSTDCXX_SUFFIX "")
    if(_CONAN_GNU_LIBSTDCXX_IS_CXX11_ABI)
        set(_CONAN_GNU_LIBSTDCXX_SUFFIX "11")
    endif()
    unset (_CONAN_GNU_LIBSTDCXX_IS_CXX11_ABI)
endmacro()


macro(detect_libcxx)
    # _CONAN_IS_LIBCXX true if LLVM libc++
    check_cxx_source_compiles("
    #include <cstddef>
    #if !defined(_LIBCPP_VERSION)
       static_assert(false);
    #endif
    int main(){}" _CONAN_IS_LIBCXX)
endmacro()


function(detect_lib_cxx LIB_CXX)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        message(STATUS "CMake-Conan: android_stl=${CMAKE_ANDROID_STL_TYPE}")
        set(${LIB_CXX} ${CMAKE_ANDROID_STL_TYPE} PARENT_SCOPE)
        return()
    endif()

    include(CheckCXXSourceCompiles)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        detect_gnu_libstdcxx()
        set(${LIB_CXX} "libstdc++${_CONAN_GNU_LIBSTDCXX_SUFFIX}" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        set(${LIB_CXX} "libc++" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        # Check for libc++
        detect_libcxx()
        if(_CONAN_IS_LIBCXX)
            set(${LIB_CXX} "libc++" PARENT_SCOPE)
            return()
        endif()

        # Check for libstdc++
        detect_gnu_libstdcxx()
        if(_CONAN_IS_GNU_LIBSTDCXX)
            set(${LIB_CXX} "libstdc++${_CONAN_GNU_LIBSTDCXX_SUFFIX}" PARENT_SCOPE)
            return()
        endif()

        # TODO: it would be an error if we reach this point
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # Do nothing - compiler.runtime and compiler.runtime_type
        # should be handled separately: https://github.com/conan-io/cmake-conan/pull/516
        return()
    else()
        # TODO: unable to determine, ask user to provide a full profile file instead
    endif()
endfunction()


function(detect_compiler COMPILER COMPILER_VERSION COMPILER_RUNTIME COMPILER_RUNTIME_TYPE)
    if(DEFINED CMAKE_CXX_COMPILER_ID)
        set(_COMPILER ${CMAKE_CXX_COMPILER_ID})
        set(_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    else()
        if(NOT DEFINED CMAKE_C_COMPILER_ID)
            message(FATAL_ERROR "C or C++ compiler not defined")
        endif()
        set(_COMPILER ${CMAKE_C_COMPILER_ID})
        set(_COMPILER_VERSION ${CMAKE_C_COMPILER_VERSION})
    endif()

    message(STATUS "CMake-Conan: CMake compiler=${_COMPILER}")
    message(STATUS "CMake-Conan: CMake compiler version=${_COMPILER_VERSION}")

    if(_COMPILER MATCHES MSVC)
        set(_COMPILER "msvc")
        string(SUBSTRING ${MSVC_VERSION} 0 3 _COMPILER_VERSION)
        # Configure compiler.runtime and compiler.runtime_type settings for MSVC
        if(CMAKE_MSVC_RUNTIME_LIBRARY)
            set(_msvc_runtime_library ${CMAKE_MSVC_RUNTIME_LIBRARY})
        else()
            set(_msvc_runtime_library MultiThreaded$<$<CONFIG:Debug>:Debug>DLL) # default value documented by CMake
        endif()

        set(_KNOWN_MSVC_RUNTIME_VALUES "")
        list(APPEND _KNOWN_MSVC_RUNTIME_VALUES MultiThreaded MultiThreadedDLL)
        list(APPEND _KNOWN_MSVC_RUNTIME_VALUES MultiThreadedDebug MultiThreadedDebugDLL)
        list(APPEND _KNOWN_MSVC_RUNTIME_VALUES MultiThreaded$<$<CONFIG:Debug>:Debug> MultiThreaded$<$<CONFIG:Debug>:Debug>DLL)

        # only accept the 6 possible values, otherwise we don't don't know to map this
        if(NOT _msvc_runtime_library IN_LIST _KNOWN_MSVC_RUNTIME_VALUES)
            message(FATAL_ERROR "CMake-Conan: unable to map MSVC runtime: ${_msvc_runtime_library} to Conan settings")
        endif()

        # Runtime is "dynamic" in all cases if it ends in DLL
        if(_msvc_runtime_library MATCHES ".*DLL$")
            set(_COMPILER_RUNTIME "dynamic")
        else()
            set(_COMPILER_RUNTIME "static")
        endif()
        message(STATUS "CMake-Conan: CMake compiler.runtime=${_COMPILER_RUNTIME}")

        # Only define compiler.runtime_type when explicitly requested
        # If a generator expression is used, let Conan handle it conditional on build_type
        if(NOT _msvc_runtime_library MATCHES "<CONFIG:Debug>:Debug>")
            if(_msvc_runtime_library MATCHES "Debug")
                set(_COMPILER_RUNTIME_TYPE "Debug")
            else()
                set(_COMPILER_RUNTIME_TYPE "Release")
            endif()
            message(STATUS "CMake-Conan: CMake compiler.runtime_type=${_COMPILER_RUNTIME_TYPE}")
        endif()

        unset(_KNOWN_MSVC_RUNTIME_VALUES)

    elseif(_COMPILER MATCHES AppleClang)
        set(_COMPILER "apple-clang")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
        list(GET VERSION_LIST 0 _COMPILER_VERSION)
    elseif(_COMPILER MATCHES Clang)
        set(_COMPILER "clang")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
        list(GET VERSION_LIST 0 _COMPILER_VERSION)
    elseif(_COMPILER MATCHES GNU)
        set(_COMPILER "gcc")
        string(REPLACE "." ";" VERSION_LIST ${CMAKE_CXX_COMPILER_VERSION})
        list(GET VERSION_LIST 0 _COMPILER_VERSION)
    endif()

    message(STATUS "CMake-Conan: [settings] compiler=${_COMPILER}")
    message(STATUS "CMake-Conan: [settings] compiler.version=${_COMPILER_VERSION}")
    if (_COMPILER_RUNTIME)
        message(STATUS "CMake-Conan: [settings] compiler.runtime=${_COMPILER_RUNTIME}")
    endif()
    if (_COMPILER_RUNTIME_TYPE)
        message(STATUS "CMake-Conan: [settings] compiler.runtime_type=${_COMPILER_RUNTIME_TYPE}")
    endif()

    set(${COMPILER} ${_COMPILER} PARENT_SCOPE)
    set(${COMPILER_VERSION} ${_COMPILER_VERSION} PARENT_SCOPE)
    set(${COMPILER_RUNTIME} ${_COMPILER_RUNTIME} PARENT_SCOPE)
    set(${COMPILER_RUNTIME_TYPE} ${_COMPILER_RUNTIME_TYPE} PARENT_SCOPE)
endfunction()


function(detect_build_type BUILD_TYPE)
    get_property(_MULTICONFIG_GENERATOR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT _MULTICONFIG_GENERATOR)
        # Only set when we know we are in a single-configuration generator
        # Note: we may want to fail early if `CMAKE_BUILD_TYPE` is not defined
        set(${BUILD_TYPE} ${CMAKE_BUILD_TYPE} PARENT_SCOPE)
    endif()
endfunction()

macro(set_conan_compiler_if_appleclang lang command output_variable)
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "AppleClang")
        execute_process(COMMAND xcrun --find ${command}
            OUTPUT_VARIABLE _xcrun_out OUTPUT_STRIP_TRAILING_WHITESPACE)
        cmake_path(GET _xcrun_out PARENT_PATH _xcrun_toolchain_path)
        cmake_path(GET CMAKE_${lang}_COMPILER PARENT_PATH _compiler_parent_path)
        if ("${_xcrun_toolchain_path}" STREQUAL "${_compiler_parent_path}")
            set(${output_variable} "")
        endif()
        unset(_xcrun_out)
        unset(_xcrun_toolchain_path)
        unset(_compiler_parent_path)
    endif()
endmacro()


macro(append_compiler_executables_configuration)
    set(_conan_c_compiler "")
    set(_conan_cpp_compiler "")
    if(CMAKE_C_COMPILER)
        set(_conan_c_compiler "\"c\":\"${CMAKE_C_COMPILER}\",")
        set_conan_compiler_if_appleclang(C cc _conan_c_compiler)
    else()
        message(WARNING "CMake-Conan: The C compiler is not defined. "
                        "Please define CMAKE_C_COMPILER or enable the C language.")
    endif()
    if(CMAKE_CXX_COMPILER)
        set(_conan_cpp_compiler "\"cpp\":\"${CMAKE_CXX_COMPILER}\"")
        set_conan_compiler_if_appleclang(CXX c++ _conan_cpp_compiler)
    else()
        message(WARNING "CMake-Conan: The C++ compiler is not defined. "
                        "Please define CMAKE_CXX_COMPILER or enable the C++ language.")
    endif()

    if(NOT "x${_conan_c_compiler}${_conan_cpp_compiler}" STREQUAL "x")
        string(APPEND PROFILE "tools.build:compiler_executables={${_conan_c_compiler}${_conan_cpp_compiler}}\n")
    endif()
    unset(_conan_c_compiler)
    unset(_conan_cpp_compiler)
endmacro()


function(detect_host_profile output_file)
    detect_os(MYOS MYOS_API_LEVEL MYOS_SDK MYOS_SUBSYSTEM MYOS_VERSION)
    detect_arch(MYARCH)
    detect_compiler(MYCOMPILER MYCOMPILER_VERSION MYCOMPILER_RUNTIME MYCOMPILER_RUNTIME_TYPE)
    detect_cxx_standard(MYCXX_STANDARD)
    detect_lib_cxx(MYLIB_CXX)
    detect_build_type(MYBUILD_TYPE)

    set(PROFILE "")
    string(APPEND PROFILE "[settings]\n")
    if(MYARCH)
        string(APPEND PROFILE arch=${MYARCH} "\n")
    endif()
    if(MYOS)
        string(APPEND PROFILE os=${MYOS} "\n")
    endif()
    if(MYOS_API_LEVEL)
        string(APPEND PROFILE os.api_level=${MYOS_API_LEVEL} "\n")
    endif()
    if(MYOS_VERSION)
        string(APPEND PROFILE os.version=${MYOS_VERSION} "\n")
    endif()
    if(MYOS_SDK)
        string(APPEND PROFILE os.sdk=${MYOS_SDK} "\n")
    endif()
    if(MYOS_SUBSYSTEM)
        string(APPEND PROFILE os.subsystem=${MYOS_SUBSYSTEM} "\n")
    endif()
    if(MYCOMPILER)
        string(APPEND PROFILE compiler=${MYCOMPILER} "\n")
    endif()
    if(MYCOMPILER_VERSION)
        string(APPEND PROFILE compiler.version=${MYCOMPILER_VERSION} "\n")
    endif()
    if(MYCOMPILER_RUNTIME)
        string(APPEND PROFILE compiler.runtime=${MYCOMPILER_RUNTIME} "\n")
    endif()
    if(MYCOMPILER_RUNTIME_TYPE)
        string(APPEND PROFILE compiler.runtime_type=${MYCOMPILER_RUNTIME_TYPE} "\n")
    endif()
    if(MYCXX_STANDARD)
        string(APPEND PROFILE compiler.cppstd=${MYCXX_STANDARD} "\n")
    endif()
    if(MYLIB_CXX)
        string(APPEND PROFILE compiler.libcxx=${MYLIB_CXX} "\n")
    endif()
    if(MYBUILD_TYPE)
        string(APPEND PROFILE "build_type=${MYBUILD_TYPE}\n")
    endif()

    if(NOT DEFINED output_file)
        set(_FN "${CMAKE_BINARY_DIR}/profile")
    else()
        set(_FN ${output_file})
    endif()

    string(APPEND PROFILE "[conf]\n")
    string(APPEND PROFILE "tools.cmake.cmaketoolchain:generator=${CMAKE_GENERATOR}\n")

    # propagate compilers via profile
    append_compiler_executables_configuration()

    if(MYOS STREQUAL "Android")
        string(APPEND PROFILE "tools.android:ndk_path=${CMAKE_ANDROID_NDK}\n")
    endif()

    message(STATUS "CMake-Conan: Creating profile ${_FN}")
    file(WRITE ${_FN} ${PROFILE})
    message(STATUS "CMake-Conan: Profile: \n${PROFILE}")
endfunction()


function(conan_profile_detect_default)
    message(STATUS "CMake-Conan: Checking if a default profile exists")
    execute_process(COMMAND ${CONAN_COMMAND} profile path default
                    RESULT_VARIABLE return_code
                    OUTPUT_VARIABLE conan_stdout
                    ERROR_VARIABLE conan_stderr
                    ECHO_ERROR_VARIABLE    # show the text output regardless
                    ECHO_OUTPUT_VARIABLE
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(NOT ${return_code} EQUAL "0")
        message(STATUS "CMake-Conan: The default profile doesn't exist, detecting it.")
        execute_process(COMMAND ${CONAN_COMMAND} profile detect
            RESULT_VARIABLE return_code
            OUTPUT_VARIABLE conan_stdout
            ERROR_VARIABLE conan_stderr
            ECHO_ERROR_VARIABLE    # show the text output regardless
            ECHO_OUTPUT_VARIABLE
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endfunction()


function(conan_install)
    cmake_parse_arguments(ARGS CONAN_ARGS ${ARGN})
    set(CONAN_OUTPUT_FOLDER ${CMAKE_BINARY_DIR}/conan)
    # Invoke "conan install" with the provided arguments
    set(CONAN_ARGS ${CONAN_ARGS} -of=${CONAN_OUTPUT_FOLDER})
    message(STATUS "CMake-Conan: conan install ${CMAKE_SOURCE_DIR} ${CONAN_ARGS} ${ARGN}")


    # In case there was not a valid cmake executable in the PATH, we inject the
    # same we used to invoke the provider to the PATH
    if(DEFINED PATH_TO_CMAKE_BIN)
        set(_OLD_PATH $ENV{PATH})
        set(ENV{PATH} "$ENV{PATH}:${PATH_TO_CMAKE_BIN}")
    endif()

    execute_process(COMMAND ${CONAN_COMMAND} install ${CMAKE_SOURCE_DIR} ${CONAN_ARGS} ${ARGN} --format=json
                    RESULT_VARIABLE return_code
                    OUTPUT_VARIABLE conan_stdout
                    ERROR_VARIABLE conan_stderr
                    ECHO_ERROR_VARIABLE    # show the text output regardless
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    if(DEFINED PATH_TO_CMAKE_BIN)
        set(ENV{PATH} "${_OLD_PATH}")
    endif()

    if(NOT "${return_code}" STREQUAL "0")
        message(FATAL_ERROR "Conan install failed='${return_code}'")
    endif()

    # the files are generated in a folder that depends on the layout used, if
    # one is specified, but we don't know a priori where this is.
    # TODO: this can be made more robust if Conan can provide this in the json output
    string(JSON CONAN_GENERATORS_FOLDER GET ${conan_stdout} graph nodes 0 generators_folder)
    cmake_path(CONVERT ${CONAN_GENERATORS_FOLDER} TO_CMAKE_PATH_LIST CONAN_GENERATORS_FOLDER)
    # message("conan stdout: ${conan_stdout}")
    message(STATUS "CMake-Conan: CONAN_GENERATORS_FOLDER=${CONAN_GENERATORS_FOLDER}")
    set_property(GLOBAL PROPERTY CONAN_GENERATORS_FOLDER "${CONAN_GENERATORS_FOLDER}")
    # reconfigure on conanfile changes
    string(JSON CONANFILE GET ${conan_stdout} graph nodes 0 label)
    message(STATUS "CMake-Conan: CONANFILE=${CMAKE_SOURCE_DIR}/${CONANFILE}")
    set_property(DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/${CONANFILE}")
    # success
    set_property(GLOBAL PROPERTY CONAN_INSTALL_SUCCESS TRUE)

endfunction()


function(conan_get_version conan_command conan_current_version)
    execute_process(
        COMMAND ${conan_command} --version
        OUTPUT_VARIABLE conan_output
        RESULT_VARIABLE conan_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(conan_result)
        message(FATAL_ERROR "CMake-Conan: Error when trying to run Conan")
    endif()

    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" conan_version ${conan_output})
    set(${conan_current_version} ${conan_version} PARENT_SCOPE)
endfunction()


function(conan_version_check)
    set(options )
    set(oneValueArgs MINIMUM CURRENT)
    set(multiValueArgs )
    cmake_parse_arguments(CONAN_VERSION_CHECK
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT CONAN_VERSION_CHECK_MINIMUM)
        message(FATAL_ERROR "CMake-Conan: Required parameter MINIMUM not set!")
    endif()
        if(NOT CONAN_VERSION_CHECK_CURRENT)
        message(FATAL_ERROR "CMake-Conan: Required parameter CURRENT not set!")
    endif()

    if(CONAN_VERSION_CHECK_CURRENT VERSION_LESS CONAN_VERSION_CHECK_MINIMUM)
        message(FATAL_ERROR "CMake-Conan: Conan version must be ${CONAN_VERSION_CHECK_MINIMUM} or later")
    endif()
endfunction()


macro(construct_profile_argument argument_variable profile_list)
    set(${argument_variable} "")
    if("${profile_list}" STREQUAL "CONAN_HOST_PROFILE")
        set(_arg_flag "--profile:host=")
    elseif("${profile_list}" STREQUAL "CONAN_BUILD_PROFILE")
        set(_arg_flag "--profile:build=")
    endif()

    set(_profile_list "${${profile_list}}")
    list(TRANSFORM _profile_list REPLACE "auto-cmake" "${CMAKE_BINARY_DIR}/conan_host_profile")
    list(TRANSFORM _profile_list PREPEND ${_arg_flag})
    set(${argument_variable} ${_profile_list})

    unset(_arg_flag)
    unset(_profile_list)
endmacro()


macro(conan_provide_dependency method package_name)
    set_property(GLOBAL PROPERTY CONAN_PROVIDE_DEPENDENCY_INVOKED TRUE)
    get_property(_conan_install_success GLOBAL PROPERTY CONAN_INSTALL_SUCCESS)
    if(NOT _conan_install_success)
        find_program(CONAN_COMMAND "conan" REQUIRED)
        conan_get_version(${CONAN_COMMAND} CONAN_CURRENT_VERSION)
        conan_version_check(MINIMUM ${CONAN_MINIMUM_VERSION} CURRENT ${CONAN_CURRENT_VERSION})
        message(STATUS "CMake-Conan: first find_package() found. Installing dependencies with Conan")
        if("default" IN_LIST CONAN_HOST_PROFILE OR "default" IN_LIST CONAN_BUILD_PROFILE)
            conan_profile_detect_default()
        endif()
        if("auto-cmake" IN_LIST CONAN_HOST_PROFILE)
            detect_host_profile(${CMAKE_BINARY_DIR}/conan_host_profile)
        endif()
        construct_profile_argument(_host_profile_flags CONAN_HOST_PROFILE)
        construct_profile_argument(_build_profile_flags CONAN_BUILD_PROFILE)
        if(EXISTS "${CMAKE_SOURCE_DIR}/conanfile.py")
            file(READ "${CMAKE_SOURCE_DIR}/conanfile.py" outfile)
            if(NOT "${outfile}" MATCHES ".*CMakeDeps.*")
                message(WARNING "Cmake-conan: CMakeDeps generator was not defined in the conanfile")
            endif()
            set(generator "")
        elseif (EXISTS "${CMAKE_SOURCE_DIR}/conanfile.txt")
            file(READ "${CMAKE_SOURCE_DIR}/conanfile.txt" outfile)
            if(NOT "${outfile}" MATCHES ".*CMakeDeps.*")
                message(WARNING "Cmake-conan: CMakeDeps generator was not defined in the conanfile. "
                        "Please define the generator as it will be mandatory in the future")
            endif()
            set(generator "-g;CMakeDeps")
        endif()
        get_property(_multiconfig_generator GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if(NOT _multiconfig_generator)
            message(STATUS "CMake-Conan: Installing single configuration ${CMAKE_BUILD_TYPE}")
            conan_install(${_host_profile_flags} ${_build_profile_flags} ${CONAN_INSTALL_ARGS} ${generator})
        else()
            message(STATUS "CMake-Conan: Installing both Debug and Release")
            conan_install(${_host_profile_flags} ${_build_profile_flags} -s build_type=Release ${CONAN_INSTALL_ARGS} ${generator})
            conan_install(${_host_profile_flags} ${_build_profile_flags} -s build_type=Debug ${CONAN_INSTALL_ARGS} ${generator})
        endif()
        unset(_host_profile_flags)
        unset(_build_profile_flags)
        unset(_multiconfig_generator)
        unset(_conan_install_success)
    else()
        message(STATUS "CMake-Conan: find_package(${ARGV1}) found, 'conan install' already ran")
        unset(_conan_install_success)
    endif()

    get_property(_conan_generators_folder GLOBAL PROPERTY CONAN_GENERATORS_FOLDER)

    # Ensure that we consider Conan-provided packages ahead of any other,
    # irrespective of other settings that modify the search order or search paths
    # This follows the guidelines from the find_package documentation
    #  (https://cmake.org/cmake/help/latest/command/find_package.html):
    #       find_package (<PackageName> PATHS paths... NO_DEFAULT_PATH)
    #       find_package (<PackageName>)

    # Filter out `REQUIRED` from the argument list, as the first call may fail
    set(_find_args_${package_name} "${ARGN}")
    list(REMOVE_ITEM _find_args_${package_name} "REQUIRED")
    if(NOT "MODULE" IN_LIST _find_args_${package_name})
        find_package(${package_name} ${_find_args_${package_name}} BYPASS_PROVIDER PATHS "${_conan_generators_folder}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        unset(_find_args_${package_name})
    endif()

    # Invoke find_package a second time - if the first call succeeded,
    # this will simply reuse the result. If not, fall back to CMake default search
    # behaviour, also allowing modules to be searched.
    if(NOT ${package_name}_FOUND)
        list(FIND CMAKE_MODULE_PATH "${_conan_generators_folder}" _index)
        if(_index EQUAL -1)
            list(PREPEND CMAKE_MODULE_PATH "${_conan_generators_folder}")
        endif()
        unset(_index)
        find_package(${package_name} ${ARGN} BYPASS_PROVIDER)
        list(REMOVE_ITEM CMAKE_MODULE_PATH "${_conan_generators_folder}")
    endif()
endmacro()


cmake_language(
    SET_DEPENDENCY_PROVIDER conan_provide_dependency
    SUPPORTED_METHODS FIND_PACKAGE
)


macro(conan_provide_dependency_check)
    set(_CONAN_PROVIDE_DEPENDENCY_INVOKED FALSE)
    get_property(_CONAN_PROVIDE_DEPENDENCY_INVOKED GLOBAL PROPERTY CONAN_PROVIDE_DEPENDENCY_INVOKED)
    if(NOT _CONAN_PROVIDE_DEPENDENCY_INVOKED)
        message(WARNING "Conan is correctly configured as dependency provider, "
                        "but Conan has not been invoked. Please add at least one "
                        "call to `find_package()`.")
        if(DEFINED CONAN_COMMAND)
            # supress warning in case `CONAN_COMMAND` was specified but unused.
            set(_CONAN_COMMAND ${CONAN_COMMAND})
            unset(_CONAN_COMMAND)
        endif()
    endif()
    unset(_CONAN_PROVIDE_DEPENDENCY_INVOKED)
endmacro()


# Add a deferred call at the end of processing the top-level directory
# to check if the dependency provider was invoked at all.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL conan_provide_dependency_check)

# Configurable variables for Conan profiles
set(CONAN_HOST_PROFILE "default;auto-cmake" CACHE STRING "Conan host profile")
set(CONAN_BUILD_PROFILE "default" CACHE STRING "Conan build profile")
set(CONAN_INSTALL_ARGS "--build=missing" CACHE STRING "Command line arguments for conan install")

find_program(_cmake_program NAMES cmake NO_PACKAGE_ROOT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_FIND_ROOT_PATH)
if(NOT _cmake_program)
    get_filename_component(PATH_TO_CMAKE_BIN "${CMAKE_COMMAND}" DIRECTORY)
    set(PATH_TO_CMAKE_BIN "${PATH_TO_CMAKE_BIN}" CACHE INTERNAL "Path where the CMake executable is")
endif()

cmake_policy(POP)
