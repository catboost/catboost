# Table of contents

   * [Multimodules](#multimodules)
       - Multimodule [DOCS](#multimodule_DOCS)
       - Multimodule [PROTO_LIBRARY](#multimodule_PROTO_LIBRARY)
       - Multimodule [PY23_LIBRARY](#multimodule_PY23_LIBRARY)
       - Multimodule [PY23_NATIVE_LIBRARY](#multimodule_PY23_NATIVE_LIBRARY)
       - Multimodule [SANDBOX_TASK](#multimodule_SANDBOX_TASK)
       - Multimodule [YQL_UDF](#multimodule_YQL_UDF)
   * [Modules](#modules)
       - Module [BENCHMARK](#module_BENCHMARK)
       - Module [BOOSTTEST](#module_BOOSTTEST)
       - Module [DEV_DLL_PROXY](#module_DEV_DLL_PROXY)
       - Module [DLL](#module_DLL)
       - Module [DLL_JAVA](#module_DLL_JAVA)
       - Module [DLL_UNIT](#module_DLL_UNIT)
       - Module [EXECTEST](#module_EXECTEST)
       - Module [FAT_OBJECT](#module_FAT_OBJECT)
       - Module [FUZZ](#module_FUZZ)
       - Module [GO_LIBRARY](#module_GO_LIBRARY)
       - Module [GO_PROGRAM](#module_GO_PROGRAM)
       - Module [GO_TEST](#module_GO_TEST)
       - Module [GTEST](#module_GTEST)
       - Module [IOS_APP](#module_IOS_APP)
       - Module [JAVA_LIBRARY](#module_JAVA_LIBRARY)
       - Module [JAVA_PROGRAM](#module_JAVA_PROGRAM)
       - Module [JTEST](#module_JTEST)
       - Module [JTEST_FOR](#module_JTEST_FOR)
       - Module [JUNIT5](#module_JUNIT5)
       - Module [LIBRARY](#module_LIBRARY)
       - Module [LINK_UNIT](#module_LINK_UNIT)
       - Module [METAQUERY](#module_METAQUERY)
       - Module [PACKAGE](#module_PACKAGE)
       - Module [PROGRAM](#module_PROGRAM)
       - Module [PY3TEST](#module_PY3TEST)
       - Module [PY3TEST_BIN](#module_PY3TEST_BIN)
       - Module [PY3_LIBRARY](#module_PY3_LIBRARY)
       - Module [PY3_PROGRAM](#module_PY3_PROGRAM)
       - Module [PYMODULE](#module_PYMODULE)
       - Module [PYTEST](#module_PYTEST)
       - Module [PYTEST_BIN](#module_PYTEST_BIN)
       - Module [PY_ANY_MODULE](#module_PY_ANY_MODULE)
       - Module [PY_LIBRARY](#module_PY_LIBRARY)
       - Module [PY_PACKAGE](#module_PY_PACKAGE)
       - Module [PY_PROGRAM](#module_PY_PROGRAM)
       - Module [RESOURCES_LIBRARY](#module_RESOURCES_LIBRARY)
       - Module [R_MODULE](#module_R_MODULE)
       - Module [TESTNG](#module_TESTNG)
       - Module [TGZ_PACKAGE](#module_TGZ_PACKAGE)
       - Module [UDF](#module_UDF)
       - Module [UDF_BASE](#module_UDF_BASE)
       - Module [UDF_LIB](#module_UDF_LIB)
       - Module [UNION](#module_UNION)
       - Module [UNITTEST](#module_UNITTEST)
       - Module [UNITTEST_FOR](#module_UNITTEST_FOR)
       - Module [UNITTEST_WITH_CUSTOM_ENTRY_POINT](#module_UNITTEST_WITH_CUSTOM_ENTRY_POINT)
       - Module [YCR_PROGRAM](#module_YCR_PROGRAM)
       - Module [YQL_PYTHON3_UDF](#module_YQL_PYTHON3_UDF)
       - Module [YQL_PYTHON_UDF](#module_YQL_PYTHON_UDF)
       - Module [YQL_PYTHON_UDF_TEST](#module_YQL_PYTHON_UDF_TEST)
       - Module [YQL_UDF_MODULE](#module_YQL_UDF_MODULE)
       - Module [YQL_UDF_TEST](#module_YQL_UDF_TEST)
       - Module [YT_UNITTEST](#module_YT_UNITTEST)
       - Module [_BASE_PROGRAM](#module__BASE_PROGRAM)
       - Module [_BASE_PY3_PROGRAM](#module__BASE_PY3_PROGRAM)
       - Module [_BASE_PYTEST](#module__BASE_PYTEST)
       - Module [_BASE_PY_PROGRAM](#module__BASE_PY_PROGRAM)
       - Module [_BASE_UNIT](#module__BASE_UNIT)
       - Module [_BASE_UNITTEST](#module__BASE_UNITTEST)
       - Module [_DLL_COMPATIBLE_LIBRARY](#module__DLL_COMPATIBLE_LIBRARY)
       - Module [_GO_BASE_UNIT](#module__GO_BASE_UNIT)
       - Module [_JAVA_LIBRARY](#module__JAVA_LIBRARY)
       - Module [_JAVA_PLACEHOLDER](#module__JAVA_PLACEHOLDER)
       - Module [_LIBRARY](#module__LIBRARY)
   * [Macros](#macros)
       - Macros [ACCELEO](#macro_ACCELEO) .. [ADD_PYTEST_BIN](#macro_ADD_PYTEST_BIN)
       - Macros [ADD_PYTEST_SCRIPT](#macro_ADD_PYTEST_SCRIPT) .. [ARCHIVE_BY_KEYS](#macro_ARCHIVE_BY_KEYS)
       - Macros [ASM_PREINCLUDE](#macro_ASM_PREINCLUDE) .. [BUILD_PLNS](#macro_BUILD_PLNS)
       - Macros [BUILTIN_PYTHON](#macro_BUILTIN_PYTHON) .. [CHECK_JAVA_DEPS](#macro_CHECK_JAVA_DEPS)
       - Macros [CLANG_EMIT_AST_CXX](#macro_CLANG_EMIT_AST_CXX) .. [CREATE_BUILDINFO_FOR](#macro_CREATE_BUILDINFO_FOR)
       - Macros [CREATE_GO_SVNVERSION_FOR](#macro_CREATE_GO_SVNVERSION_FOR) .. [DARWIN_SIGNED_RESOURCE](#macro_DARWIN_SIGNED_RESOURCE)
       - Macros [DARWIN_STRINGS_RESOURCE](#macro_DARWIN_STRINGS_RESOURCE) .. [DISABLE](#macro_DISABLE)
       - Macros [DLL_FOR](#macro_DLL_FOR) .. [ENDIF](#macro_ENDIF)
       - Macros [ENV](#macro_ENV) .. [EXTRALIBS](#macro_EXTRALIBS)
       - Macros [EXTRALIBS_STATIC](#macro_EXTRALIBS_STATIC) .. [FUZZ_DICTS](#macro_FUZZ_DICTS)
       - Macros [FUZZ_OPTS](#macro_FUZZ_OPTS) .. [GO_CGO1_FLAGS](#macro_GO_CGO1_FLAGS)
       - Macros [GO_CGO2_FLAGS](#macro_GO_CGO2_FLAGS) .. [GO_XTEST_SRCS](#macro_GO_XTEST_SRCS)
       - Macros [GRPC](#macro_GRPC) .. [IOS_APP_COMMON_FLAGS](#macro_IOS_APP_COMMON_FLAGS)
       - Macros [IOS_APP_SETTINGS](#macro_IOS_APP_SETTINGS) .. [JOIN_SRCS](#macro_JOIN_SRCS)
       - Macros [JOIN_SRCS_GLOBAL](#macro_JOIN_SRCS_GLOBAL) .. [LLVM_COMPILE_C](#macro_LLVM_COMPILE_C)
       - Macros [LLVM_COMPILE_CXX](#macro_LLVM_COMPILE_CXX) .. [MAPKIT_ENABLE_WHOLE_ARCHIVE](#macro_MAPKIT_ENABLE_WHOLE_ARCHIVE)
       - Macros [MAPSMOBI_SRCS](#macro_MAPSMOBI_SRCS) .. [NO_BUILD_IF](#macro_NO_BUILD_IF)
       - Macros [NO_CHECK_IMPORTS](#macro_NO_CHECK_IMPORTS) .. [NO_NEED_CHECK](#macro_NO_NEED_CHECK)
       - Macros [NO_OPTIMIZE](#macro_NO_OPTIMIZE) .. [NO_UTIL](#macro_NO_UTIL)
       - Macros [NO_WERROR](#macro_NO_WERROR) .. [PARTITIONED_RECURSE_ROOT_RELATIVE](#macro_PARTITIONED_RECURSE_ROOT_RELATIVE)
       - Macros [PEERDIR](#macro_PEERDIR) .. [PYTHON2_MODULE](#macro_PYTHON2_MODULE)
       - Macros [PYTHON3_ADDINCL](#macro_PYTHON3_ADDINCL) .. [PY_REGISTER](#macro_PY_REGISTER)
       - Macros [PY_SRCS](#macro_PY_SRCS) .. [RUN_ANTLR](#macro_RUN_ANTLR)
       - Macros [RUN_ANTLR4](#macro_RUN_ANTLR4) .. [SETUP_RUN_PYTHON](#macro_SETUP_RUN_PYTHON)
       - Macros [SET_APPEND](#macro_SET_APPEND) .. [SRCDIR](#macro_SRCDIR)
       - Macros [SRCS](#macro_SRCS) .. [SRC_C_AVX2](#macro_SRC_C_AVX2)
       - Macros [SRC_C_SSE2](#macro_SRC_C_SSE2) .. [TAG](#macro_TAG)
       - Macros [TASKLET](#macro_TASKLET) .. [UBERJAR_HIDE_EXCLUDE_PATTERN](#macro_UBERJAR_HIDE_EXCLUDE_PATTERN)
       - Macros [UBERJAR_HIDING_PREFIX](#macro_UBERJAR_HIDING_PREFIX) .. [USE_PYTHON3](#macro_USE_PYTHON3)
       - Macros [USE_RECIPE](#macro_USE_RECIPE) .. [YABS_SERVER_BUILD_YSON_INDEX](#macro_YABS_SERVER_BUILD_YSON_INDEX)
       - Macros [YABS_SERVER_PREPARE_QXL_FROM_SANDBOX](#macro_YABS_SERVER_PREPARE_QXL_FROM_SANDBOX) .. [_BUILDWITH_CYTHON_C_DEP](#macro__BUILDWITH_CYTHON_C_DEP)
       - Macros [_BUILDWITH_CYTHON_C_H](#macro__BUILDWITH_CYTHON_C_H) .. [_GENERATE_PY_EVS_INTERNAL](#macro__GENERATE_PY_EVS_INTERNAL)
       - Macros [_GENERATE_PY_PROTOS_INTERNAL](#macro__GENERATE_PY_PROTOS_INTERNAL) .. [_GO_PROCESS_SRCS](#macro__GO_PROCESS_SRCS)
       - Macros [_GO_PROTOC_PLUGIN_ARGS_BASE](#macro__GO_PROTOC_PLUGIN_ARGS_BASE) .. [_MAPKIT_CPP_PROTO_CMD](#macro__MAPKIT_CPP_PROTO_CMD)
       - Macros [_MX_BIN_TO_INFO](#macro__MX_BIN_TO_INFO) .. [_PY_EVLOG_CMD_BASE](#macro__PY_EVLOG_CMD_BASE)
       - Macros [_PY_EVLOG_CMD_INTERNAL](#macro__PY_EVLOG_CMD_INTERNAL) .. [_SRC____C](#macro__SRC____C)
       - Macros [_SRC____S](#macro__SRC____S) .. [_SRC____external](#macro__SRC____external)
       - Macros [_SRC____f](#macro__SRC____f) .. [_SRC____lpp](#macro__SRC____lpp)
       - Macros [_SRC____m](#macro__SRC____m) .. [_SRC____rl6](#macro__SRC____rl6)
       - Macros [_SRC____s](#macro__SRC____s) .. [_SRC____yasm](#macro__SRC____yasm)
       - Macros [_SRC____ydl](#macro__SRC____ydl) .. [_UNITTEST](#macro__UNITTEST)
       - Macros [_YCR_GENERATE_CONFIGS](#macro__YCR_GENERATE_CONFIGS) .. [_YTEST](#macro__YTEST)
## Multimodules <a name="multimodules"></a>

###### Multimodule [DOCS][]() <a name="multimodule_DOCS"></a>
Definition of the documentation multimodule.

The output artifact is docs.tar.gz with statically generated site (using mkdocs as builder) when built directly
When PEERDIRed from other DOCS() behaves like a library (supplying own content and dependencies to build target).
Is only used together with the macros DOCS\_DIR(), DOCS\_CONFIG(), DOCS\_VARS().
@see: [DOCS\_DIR()](#macro\_DOCS\_DIR), [DOCS\_CONFIG()](#macro\_DOCS\_CONFIG), [DOCS\_VARS()](#macro\_DOCS\_VARS).

###### Multimodule [PROTO\_LIBRARY][]() <a name="multimodule_PROTO_LIBRARY"></a>
Definition of multimodule that builds various variants of libraries.
The particular variant is selected based on where PEERDIR to PROTO\_LIBRARY comes from.
Now supported 5 variants: C++, Java, Python 2.x, Python 3.x and Go
Python versions emit C++ code in addition to Python as optimization
PROTO\_LIBRARY also supports emission of GRPC code if GRPC() macro is specified.
Documentation: https://wiki.yandex-team.ru/yatool/proto\_library/
See: [GRPC()](#macro\_GRPC), [OPTIMIZE\_PY\_PROTOS()](#macro\_OPTIMIZE\_PY\_PROTOS)

###### Multimodule [PY23\_LIBRARY][]([name]) <a name="multimodule_PY23_LIBRARY"></a>
Build PY\_LIBRARY or PY3\_LIBRARY depending on incoming PEERDIR
Direct build or build by RECURSE creates both variants
For more information read https://wiki.yandex-team.ru/arcadia/python/pysrcs

###### Multimodule PY23\_LIBRARY([name]) <a name="multimodule_PY23_NATIVE_LIBRARY"></a>
Build LIBRARY compatible with either Puthon 2.x or Python 3.x depending on incoming PEERDIR.
This multimodule doesn't depend on Arcadia Python binary build. It is intended only for C++ code and cannot contain PY\_SRCS and USE\_PYTHON macros.
This is needed to steer proper headers selection in PYMODULE builds.
For more information read https://wiki.yandex-team.ru/arcadia/python/pysrcs/#pysrcssrcsipy23nativelibrary

###### Multimodule [SANDBOX\_TASK][]([Name]) <a name="multimodule_SANDBOX_TASK"></a>
Multimodule describing Sandbox task (Python code that can be executed by Sandbox system)

As a final target this multimodule builds Sandbox binary task. It may PEERDIR other SANDBOX\_TASKs as libraries.
As PEERDIR target it works like normal PY\_LIBRARY with predefined dependency on Sanbox SDK

Currenly Sandbox supports only Python 2.x, so both variants will be
compatible only with Python 2.x modules and will select mutimodule variants accordingly.
Documentation: https://wiki.yandex-team.ru/sandbox/tasks/binary

###### Multimodule [YQL\_UDF][](name) <a name="multimodule_YQL_UDF"></a>
Definition of multimodule which is YQL\_UDF\_MODULE when built directly or referred by BUNDLE and DEPENDS
and static LIBRARY is used by PEERDIR

## Modules <a name="modules"></a>

###### Module [BENCHMARK][]([benchmarkname]) <a name="module_BENCHMARK"></a>
Benchmark test based on library library/testing/benchmark
For more details see: https://wiki.yandex-team.ru/yatool/test/#zapuskbenchmark

###### Module [BOOSTTEST][]([name]) _#deprecated_ <a name="module_BOOSTTEST"></a>
Test module based on boost/test/unit\_test.hpp
As with entire boost library usage of this technology is deprecated in Arcadia.
No new module of this type should be introduced unless it is explicitly approved by C++ committee

###### Module [DEV\_DLL\_PROXY][]() _# deprecated_ <a name="module_DEV_DLL_PROXY"></a>
The use of this module is strictly prohibited!!!
This is a temporary and project-specific solution.

###### Module [DLL][](name major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_DLL"></a>
Definition of the project is a dynamic library.
1. major\_ver and minor\_ver must be integers.
2. EXPORTS allows you to explicitly specify the list of exported functions.
3. PREFIX allows you to change the prefix of the output file (default DLL has the prefix "lib").

DLL cannot participate in linking to programs but can be used from Java or as final artifact (packaged and deployed).

###### Module [DLL\_JAVA][]() <a name="module_DLL_JAVA"></a>
DLL built using swig for Java produces dynamic library and a .jar.
Dynamic library is treated the same as in the case of PEERDIR from Java to DLL.
.jar goes on the classpath.

Documentation: https://wiki.yandex-team.ru/yatool/java/#integracijascpp/pythonsborkojj

###### Module [DLL\_UNIT][] _# internal_ <a name="module_DLL_UNIT"></a>
Base module for all dynamically linked libraries as final artifacts.
Contains all general logic for such kind of modules. Supports versioning and export files
Cannot participate in linking to programs, intended to be used as final artifact (packaged and deployed).

###### Module [EXECTEST][]() <a name="module_EXECTEST"></a>
Module definition of generic test that executes a binary
Use macro RUN to specify binary to run

@example:
    EXECTEST()

    OWNER(g:yatool)

    RUN(
       cat input.txt
    )

    DATA(
        arcadia/devtools/ya/test/tests/exectest/data
    )

    DEPENDS(
       devtools/dummy\_arcadia/cat
    )
    TEST\_CWD(devtools/ya/test/tests/exectest/data)

    END()

More examples: https://wiki.yandex-team.ru/yatool/test/#exec-testy
@see: [RUN()](#macro\_RUN)

###### Module [FAT\_OBJECT][]() <a name="module_FAT_OBJECT"></a>
Definition of the "fat" object module. It will contain all its transitive dependencies reachable by PEERDIRs:
static libraries, local (from own SRCS) and global (from peers') object files.

Designed for use in XCode projects for iOS.
It is recommended not to specify the name

###### Module [FUZZ][]() <a name="module_FUZZ"></a>
In order to start using Fuzzing in Arcadia, you need to create a FUZZ module with the implementation of the function LLVMFuzzerTestOneInput()
(example: https://a.yandex-team.ru/arc/trunk/arcadia/contrib/libs/re2/re2/fuzzing/re2\_fuzzer.cc?rev=2919463#L58)
This module should be reachable by RECURSE from /autocheck project
in Arcadia supported AFL and Libfuzzer on top of a single interface, but the automatic fuzzing still works only through Libfuzzer.

Documentation: https://wiki.yandex-team.ru/yatool/fuzzing/

###### Module [GO\_LIBRARY][]([name]) <a name="module_GO_LIBRARY"></a>
Go library module definition.
Compile Go module as a library suitable for PEERDIR from other Go modules
Will select Go implementation on PEERDIR to PROTO\_LIBRARY

###### Module [GO\_PROGRAM][]([name]) <a name="module_GO_PROGRAM"></a>
Go program module definition
Compile and link Go module as an executable program
Will select Go implementation on PEERDIR to PROTO\_LIBRARY

###### Module [GO\_TEST][]([name]) <a name="module_GO_TEST"></a>
Go test module definition
Compile and link Go module as a test suitable for running with Arcadia testing support
All usual testing support macros like DATA, DEPENDS, SIZE, REQUIREMENTS etc. are supported
Will select Go implementation on PEERDIR to PROTO\_LIBRARY

###### Module [GTEST][]([name]) <a name="module_GTEST"></a>
Definition of the test module based on gtest (contrib/libs/gtest contrib/libs/gmock)

Use publict documentation on gtest for details
Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Module [IOS\_APP][]() <a name="module_IOS_APP"></a>
iOS application module definition

###### Module [JAVA\_LIBRARY][]() <a name="module_JAVA_LIBRARY"></a>
Documentation: https://wiki.yandex-team.ru/yatool/java/

###### Module [JAVA\_PROGRAM][]() <a name="module_JAVA_PROGRAM"></a>
JAVA\_PROGRAM() - module for describing java programs.
Output artifacts: .jar and directory with all the jar to the classpath of the formation.

Documentation: https://wiki.yandex-team.ru/yatool/java/

###### Module [JTEST][]() <a name="module_JTEST"></a>
Java tests module based on JUnit 4 framework

If requested, build system will scan the source code of the module for the presence of junit tests and run them.
Output artifacts: a jar, a directory of exhaust tests(if required run the tests) - test logs, system logs testiranja, temporary files, tests, etc.

Documentation: https://wiki.yandex-team.ru/yatool/test/#testynajava

###### Module [JTEST\_FOR][](ModuleDir) <a name="module_JTEST_FOR"></a>
Java tests module based on JUnit 4 framework for specified library or program

In contrast to the JTEST, the build system will scan for the presence of the test sources of the module in ModuleDir . As ModuleDir should contain JAVA\_PROGRAM or JAVA\_LIBRARY . JTEST\_FOR also can have its own source, in this case they will be compiled and added to the classpath of a test run.
Output artifacts: a jar, a directory of exhaust tests(if requested tests are run).

Documentation: https://wiki.yandex-team.ru/yatool/test/#testynajava

###### Module [JUNIT5][]() <a name="module_JUNIT5"></a>
Java tests module based on JUnit 5 framework

If requested, build system will scan the source code of the module for the presence of junit tests and run them.
Output artifacts: a jar, a directory of exhaust tests(if required run the tests) - test logs, system logs testiranja, temporary files, tests, etc.

Documentation: https://wiki.yandex-team.ru/yatool/test/#testynajava

###### Module [LIBRARY][]() <a name="module_LIBRARY"></a>
Definition of the static library module.
This is C++ library, and it selects peers from multimodules accordingly

It is recommended not to specify the name.

###### Module [LINK\_UNIT][]  _# internal_ <a name="module_LINK_UNIT"></a>
The base of all linkable modules: programs, DLLs etc. Describes common linking logic

###### Module [METAQUERY][]() <a name="module_METAQUERY"></a>
Project Definition - KIWI Meta query. (Objected)
https://wiki.yandex-team.ru/robot/manual/kiwi/techdoc/design/metaquery/

###### Module [PACKAGE][](name) <a name="module_PACKAGE"></a>
Module collects what is described directly inside it, builds and collects all its transitively available PEERDIRs.
As a result, build directory of the project gets the structure of the accessible part of Arcadia, where the build result of each PEERDIR is placed to relevant Arcadia subpath.

Is only used together with the macros FILES(), PACK(), COPY(), CONFIGURE\_FILE(), FROM\_SANDBOX(), PEERDIR().

Documentation: https://wiki.yandex-team.ru/yatool/large-data/

###### Module [PROGRAM][]([progname]) <a name="module_PROGRAM"></a>
Regular program module.
If name is not specified it will be generated from the name of the containing project directory.

###### Module [PY3TEST][]([name]) <a name="module_PY3TEST"></a>
Definition of the test module for Python 3.x based on py.test

This module is compatible only with PYTHON3-tagged modules and slects peers from multimodules accordingly
This module is only compatible with Arcadia Python (to avoid tests duplication from Python2/3-tests). For non-Arcadia python use PYTEST

Documentation: https://wiki.yandex-team.ru/yatool/test/#testynapytest
Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Module [PY3TEST\_BIN][]: \_BASE\_PY3\_PROGRAM <a name="module_PY3TEST_BIN"></a>
Not documented yet.

###### Module [PY3\_LIBRARY][]() <a name="module_PY3_LIBRARY"></a>
Python 3.x binary library. Builds sources from PY\_SRCS to data suitable for PY\_PROGRAM
Adds dependencies to Python 2.x runtime library from Arcadia.
This module is only compatible with PYTHON3-tagged modules and selects those from multimodules
This module is only compatible with Arcadia Python build
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/

###### Module [PY3\_PROGRAM][]([progname]) <a name="module_PY3_PROGRAM"></a>
Python 3.x binary program. Links all Python 3.x libraries and Python 3.x interpreter into itself to form regular executable.
If name is not specified it will be generated from the name of the containing project directory.
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/
This only compatible with PYTHON3-tagged modules and selects those from multimodules

###### Module [PYMODULE][](name major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_PYMODULE"></a>
Definition of the Python external module for Python2 and any system Python
1. major\_ver and minor\_ver must be integers.
2. The resulting .so will have the prefix "lib".
3. Processing EXPORTS and PREFIX is the same as for DLL module

Note: this module will always PEERDIR Python2 version of PY23\_NATIVE\_LIBRARY
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/

###### Module [PYTEST][]([name]) <a name="module_PYTEST"></a>
Definition of the test module for Python 2.x based on py.test

This module is compatible only with PYTHON2-tagged modules and slects peers from multimodules accordingly
This module is compatible with non-Arcadia Python

Documentation: https://wiki.yandex-team.ru/yatool/test/#testynapytest
Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Module [PYTEST\_BIN][]() _#deprecated_ <a name="module_PYTEST_BIN"></a>
Same as PYTEST. Don't use this

###### Module [PY\_ANY\_MODULE][](name major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_PY_ANY_MODULE"></a>
Definition of the Python external module for any arcadia and system Python.
1. major\_ver and minor\_ver must be integers.
2. The resulting .so will have the prefix "lib".
3. Processing EXPORTS and PREFIX is the same as for DLL module

Note: Use PYTHON2\_MODULE()/PYTHON3\_MODULE in order to PEERDIR proper version of PY23\_NATIVE\_LIBRARY
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/

###### Module [PY\_LIBRARY][]() <a name="module_PY_LIBRARY"></a>
Python 2.x binary library. Builds sources from PY\_SRCS to data suitable for PY\_PROGRAM
Adds dependencies to Python 2.x runtime library from Arcadia.
This module is only compatible with PYTHON2-tagged modules and selects those from multimodules
This module is only compatible with Arcadia Python build
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/

###### Module [PY\_PACKAGE][]: UNION <a name="module_PY_PACKAGE"></a>
Not documented yet.

###### Module [PY\_PROGRAM][]([progname]) <a name="module_PY_PROGRAM"></a>
Python 2.x binary program. Links all Python 2.x libraries and Python 2.x interpreter into itself to form regular executable.
If name is not specified it will be generated from the name of the containing project directory.
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/
This only compatible with PYTHON2-tagged modules and selects those from multimodules

###### Module [RESOURCES\_LIBRARY][]() <a name="module_RESOURCES_LIBRARY"></a>
Definition of a module that brings its content from external source (sandbox)
via DECLARE\_EXTERNAL\_RESOURCE macro
This can participate in PEERDIRs of others as library but it cannot have own sources

It is recommended not to specify the name.
@see: [DECLARE\_EXTERNAL\_RESOURCE()](#macro\_DECLARE\_EXTERNAL\_RESOURCE)

###### Module [R\_MODULE][](name major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_R_MODULE"></a>
Definition of the external module for R language
1. major\_ver and minor\_ver must be integers.
2. The resulting .so will have the prefix "lib".
3. Processing EXPORTS and PREFIX is the same as for DLL module

###### Module [TESTNG][]() <a name="module_TESTNG"></a>
Java tests module based on TestNG framework

If requested, build system will scan the source code of the module for the presence of junit tests and run them.
Output artifacts: a jar, a directory of exhaust tests(if required run the tests) - test logs, system logs testiranja, temporary files, tests, etc.

Documentation: https://wiki.yandex-team.ru/yatool/test/#testynajava

###### Module [TGZ\_PACKAGE][]() _# internal deprecated_ <a name="module_TGZ_PACKAGE"></a>
Don't use this, use PACKAGE module with PACK() macro instead
@see: [PACKAGE](#module\_PACKAGE), [PACK()](#macro\_PACK)

###### Module [UDF][](name [EXPORTS symlist\_file] [PREFIX prefix]) _# deprecated_ <a name="module_UDF"></a>
Definition of the KiWi UDF module
Processing EXPORTS and PREFIX is the same as for DLL
https://wiki.yandex-team.ru/robot/manual/kiwi/userguide/#polzovatelskiefunkciiudftriggerykwcalc

###### Module [UDF\_BASE][]: DLL\_UNIT <a name="module_UDF_BASE"></a>
Not documented yet.

###### Module [UDF\_LIB][]([name]) _# deprecated_ <a name="module_UDF_LIB"></a>
The LIBRARY module for KiWi UDF, so has default PEERDIR to yweb/robot/kiwi/kwcalc/udflib
It is recommended not to specify a name

###### Module [UNION][](name) <a name="module_UNION"></a>
Module is able to collect what is described directly inside it, remember PEERDIRs listed in it, go there and build them.
However PEERDIRs build results are not used themselves, semantics of PEERDIR is the same as for LIBRARY.
When built directly PEEDIRs won't be built unless it is built via DEPENDS macro, which treats the UNION specially.

Is only used together with the macros FILES(), PEERDIR(), COPY(), FROM\_SANDBOX().

Documentation: https://wiki.yandex-team.ru/yatool/large-data/

###### Module [UNITTEST][]([name]) <a name="module_UNITTEST"></a>
Unit test module based on library/unittest

It is recommended not to specify the name
Documentation: https://wiki.yandex-team.ru/yatool/test/#opisanievya.make1

###### Module [UNITTEST\_FOR][](path/to/lib) <a name="module_UNITTEST_FOR"></a>
Convenience module declaration - extension of UNITTEST module
Module UNITTEST\_FOR automatically makes SRCDIR + ADDINCL + PEERDIR on path/to/lib.
path/to/lib is the path to the directory with the LIBRARY project.
Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Module [UNITTEST\_WITH\_CUSTOM\_ENTRY\_POINT][]([name]) <a name="module_UNITTEST_WITH_CUSTOM_ENTRY_POINT"></a>
Generic unit test module

###### Module [YCR\_PROGRAM][]([progname]) <a name="module_YCR_PROGRAM"></a>
yacare-specific program module. Generates yacare configs in addition to producing the program
If name is not specified it will be generated from the name of the containing project directory.

###### Module [YQL\_PYTHON3\_UDF][](name) <a name="module_YQL_PYTHON3_UDF"></a>
Definition of the extension module for YQL with Python 3.x UDF (User Defined Function YQL)
Documentation: https://yql.yandex-team.ru/docs/yt/udf/python/

###### Module [YQL\_PYTHON\_UDF][](name) <a name="module_YQL_PYTHON_UDF"></a>
Definition of the extension module for YQL with Python 2.x UDF (User Defined Function YQL)
https://yql.yandex-team.ru/docs/yt/udf/python/

###### Module [YQL\_PYTHON\_UDF\_TEST][](name) <a name="module_YQL_PYTHON_UDF_TEST"></a>
Definition of the Python test for UDF (User Defined Function YQL)
Documentation: https://yql.yandex-team.ru/docs/yt/udf/python/
@example: https://a.yandex-team.ru/arc/trunk/arcadia/yql/udfs/test/simple/ya.make

###### Module [YQL\_UDF\_MODULE][](name) <a name="module_YQL_UDF_MODULE"></a>
Definition of the extension module for YQL with C++ UDF (User Defined Function YQL)
https://yql.yandex-team.ru/docs/yt/udf/cpp/

###### Module [YQL\_UDF\_TEST][]([name]) <a name="module_YQL_UDF_TEST"></a>
Definition of the module to test YQL C++ UDF.
Documentation: https://yql.yandex-team.ru/docs/yt/libraries/testing/
Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Module [YT\_UNITTEST][]([name]) <a name="module_YT_UNITTEST"></a>
YT Unit test module based on mapreduce/yt/library/utlib

###### Module [\_BASE\_PROGRAM][]  _# internal_ <a name="module__BASE_PROGRAM"></a>
The base of all programs. It adds dependencies to make final artefact complete and runnable

###### Module \_BASE\_PY\_PROGRAM _#internal_ <a name="module__BASE_PY3_PROGRAM"></a>
The base module for all Python 3.x binary programs. Adds linking logic, relevant module properties and
dependency on Python 3.x interpreter. Also adds import tests on all sources including libraries.
Links all Python 3.x libraries and Python 3.x interpreter into itself to form regular executable.

Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/
This only compatible with PYTHON3-tagged modules and selects those from multimodules

###### Module [\_BASE\_PYTEST][]  _# internal_ <a name="module__BASE_PYTEST"></a>
Base logic of Python 2.x py.test modules: common module properties and dependencies

###### Module [\_BASE\_PY\_PROGRAM][] _#internal_ <a name="module__BASE_PY_PROGRAM"></a>
The base module for all Python 2.x binary programs. Adds linking logic, relevant module properties and
dependency on Python 2.x interpreter. Also adds import tests on all sources including libraries.
Links all Python 2.x libraries and Python 2.x interpreter into itself to form regular executable.

Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs/
This only compatible with PYTHON2-tagged modules and selects those from multimodules

###### Module [\_BASE\_UNIT][]  _# internal_ <a name="module__BASE_UNIT"></a>
The base of all buildable modules describing commong logic for all modules.
To avoid surprises all buildable modules are better to be inherited from it or its descendants

###### Module [\_BASE\_UNITTEST][]  _# internal_ <a name="module__BASE_UNITTEST"></a>
Module with base logic for all unit-test modules: it makes code runnable as unit-test by Arcadia testing machiery

###### Module [\_DLL\_COMPATIBLE\_LIBRARY][] _# internal_ <a name="module__DLL_COMPATIBLE_LIBRARY"></a>
Base module to place DLLs into multimodules back to back with libraries
In order to function properly all modules in multimodule shall have the
same set of arguments. So this module is just library that accepts but
ignores all DLL arguments

###### Module [\_GO\_BASE\_UNIT][]: \_BASE\_UNIT _#internal_ <a name="module__GO_BASE_UNIT"></a>
Not documented yet.

###### Module [\_JAVA\_LIBRARY][]() _#internal_ <a name="module__JAVA_LIBRARY"></a>
\_JAVA\_LIBRARY() is a module for creating a .jar file using non-Java code (generators etc.)

###### Module [\_JAVA\_PLACEHOLDER][]: \_BASE\_UNIT _#internal_ <a name="module__JAVA_PLACEHOLDER"></a>
Not documented yet.

###### Module [\_LIBRARY][] _# internal_ <a name="module__LIBRARY"></a>
Base module definition for all libraries
Contains basic logic like module properties, default variable values etc.
All libraries similar to C++-libraries should be inherited from it

## Macros <a name="macros"></a>

###### Macro [ACCELEO][] <a name="macro_ACCELEO"></a>
Not documented yet.

###### Macro [ADDINCL][]([GLOBAL dir]\* dirlist)  _# builtin_ <a name="macro_ADDINCL"></a>
The macro adds the -I<library path> flag to the source compilation flags of the current project.
@params:
GLOBAL - extends the search for headers (-I) on the dependent projects

###### Macro [ADDINCLSELF][] <a name="macro_ADDINCLSELF"></a>
Not documented yet.

###### Macro [ADD\_CHECK][] <a name="macro_ADD_CHECK"></a>
Not documented yet.

###### Macro [ADD\_CHECK\_PY\_IMPORTS][] <a name="macro_ADD_CHECK_PY_IMPORTS"></a>
Not documented yet.

###### Macro [ADD\_COMPILABLE\_TRANSLATE][](Dict Name Options...) <a name="macro_ADD_COMPILABLE_TRANSLATE"></a>
Generate translation dictionary code to transdict.LOWER(Name).cpp that will than be compiled into library

###### Macro [ADD\_COMPILABLE\_TRANSLIT][](TranslitTable NGrams Name Options...) <a name="macro_ADD_COMPILABLE_TRANSLIT"></a>
Generate transliteration dictionary code
This will emit both translit, untranslit and ngrams table codes those will be than further compiled into library

###### Macro [ADD\_DLLS\_TO\_JAR][]() <a name="macro_ADD_DLLS_TO_JAR"></a>
Not documented yet.

###### Macro [ADD\_PERL\_MODULE][](Dir ModuleName) <a name="macro_ADD_PERL_MODULE"></a>
Add dependency on specified Perl module to the library

###### Macro [ADD\_PYTEST\_BIN][] <a name="macro_ADD_PYTEST_BIN"></a>
Not documented yet.

###### Macro [ADD\_PYTEST\_SCRIPT][] <a name="macro_ADD_PYTEST_SCRIPT"></a>
Not documented yet.

###### Macro [ADD\_TEST][] <a name="macro_ADD_TEST"></a>
Not documented yet.

###### Macro [ADD\_WAR][]([INCLUDE Path Pattern...] [EXCLUDE Path Pattern...] [INCLUDE\_DEFAULT Pattern...] [EXCLUDE\_DEFAULT Pattern...]) <a name="macro_ADD_WAR"></a>
Pack Java module (JAVA\_PROGRAM or JAVA\_LIBRARY) into war.
Parameters control what should be added or shouldn't be added to war archive.
Documentation: https://wiki.yandex-team.ru/yatool/java/war

###### Macro [ADD\_YTEST][] <a name="macro_ADD_YTEST"></a>
Not documented yet.

###### Macro [ALLOCATOR][](Alloc)  _# Default: LF_ <a name="macro_ALLOCATOR"></a>
Set allocator implementation for a program
This may only be specified for programs. Use inside LIBRARY ALLOCATOR() END() leads to errors and stop the build.
Available allocators are: "LF", "LF\_YT", "LF\_DBG", "J", "B", "BS", "BM", "C", "GOOGLE", "LOCKLESS", "SYSTEM", "FAKE"
  - LF - lfalloc (https://a.yandex-team.ru/arc/trunk/arcadia/library/lfalloc)
  - LF\_YT -  Allocator selection for YT (https://a.yandex-team.ru/arc/trunk/arcadia/library/lfalloc/yt/ya.make)
  - LF\_DBG -  Debug allocator selection(https://a.yandex-team.ru/arc/trunk/arcadia/library/lfalloc/dbg/ya.make)
  - J - The JEMalloc allocator (https://a.yandex-team.ru/arc/trunk/arcadia/library/malloc/jemalloc)
  - B - The balloc allocator named Pyotr Popov and Anton Samokhvalov
      - Discussion: https://ironpeter.at.yandex-team.ru/replies.xml?item\_no=126
      - Code: https://a.yandex-team.ru/arc/trunk/arcadia/library/balloc
  - BS - The balloc allocator with leak sanitizer (https://a.yandex-team.ru/arc/trunk/arcadia/library/balloc/sanitize)
  - BM - The balloc for market (agri@ commits from july 2018 till november 2018 saved)
  - C - Like B, but can be disabled for each thread to LF or SYSTEM one (B can be disabled only to SYSTEM)
  - GOOGLE -  Google TCMalloc (https://a.yandex-team.ru/arc/trunk/arcadia/library/malloc/galloc)
  - LOCKLESS - Allocator based upon lockless queues (https://a.yandex-team.ru/arc/trunk/arcadia/library/malloc/lockless)
  - SYSTEM - Use target system allocator
  - FAKE - Don't link with any allocator

More about the allocator in Arcadia: https://wiki.yandex-team.ru/arcadia/allocators/

###### Macro [ALL\_SRCS][]([GLOBAL] filenames...) <a name="macro_ALL_SRCS"></a>
Make all source files listed as GLOBAL or not depending on the keyword GLOBAL
Call to ALL\_SRCS macro is equivalent to call to GLOBAL\_SRCS macro when GLOBAL keyword is specified
as the first argument and is equivalent to call to SRCS macro otherwise.

@example:
Consider the file to ya.make:
```
    LIBRARY()
    SET(MAKE\_IT\_GLOBAL GLOBAL)
    ALL\_SRCS(${MAKE\_IT\_GLOBAL} foo.cpp bar.cpp)
    END()
```
@see: [GLOBAL\_SRCS()](#macro\_GLOBAL\_SRCS), [SRCS()](#macro\_SRCS)

###### Macro [ANNOTATION\_PROCESSOR][](processors...) <a name="macro_ANNOTATION_PROCESSOR"></a>
The macro is in development.
Used to specify annotation processors to build JAVA\_PROGRAM() and JAVA\_LIBRARY()

###### Macro [ARCHIVE][](NAME, archive\_name files...) <a name="macro_ARCHIVE"></a>
To add data (resources, arbitrary files)
Example: https://wiki.yandex-team.ru/yatool/howtowriteyamakefiles/#a1ispolzujjtekomanduarchive

###### Macro [ARCHIVE\_ASM][](NAME archive\_name files...) <a name="macro_ARCHIVE_ASM"></a>
Similar to the macro ARCHIVE, but ARCHIVE\_ASM:
1. works faster and it is better to use for large files.
2. Different syntax (see examples in codesearch or users/pg/tests/archive\_test)

###### Macro [ARCHIVE\_BY\_KEYS][](NAME="", KEYS="", DONTCOMPRESS?"-p":"", Files...) <a name="macro_ARCHIVE_BY_KEYS"></a>
Not documented yet.

###### Macro [ASM\_PREINCLUDE][](AsmFiles...) <a name="macro_ASM_PREINCLUDE"></a>
Supply additional .asm files to all assembler calls within a module

###### Macro [BASE\_CODEGEN][](tool\_path prefix) <a name="macro_BASE_CODEGEN"></a>
Generator ${prefix}.cpp + ${prefix}.h files based on ${prefix}.in

###### Macro [BUILDWITH\_CYTHON\_C][](Src Options...) <a name="macro_BUILDWITH_CYTHON_C"></a>
Generates .c file from .pyx

###### Macro [BUILDWITH\_CYTHON\_CPP][](Src Options...) <a name="macro_BUILDWITH_CYTHON_CPP"></a>
Generates. cpp file from .pyx

###### Macro [BUILDWITH\_RAGEL6][](Src Options...) <a name="macro_BUILDWITH_RAGEL6"></a>
Compile .rl file using Ragel6

###### Macro BUILD\_CB(cbmodel cbname) <a name="macro_BUILD_CATBOOST"></a>
Generate catboost model and access code
cbmodel - CatBoost model file name (\*.cmb)
cbname - name for a variable (of NCatboostCalcer::TCatboostCalcer type) to be available in CPP code
CatBoost specific macro

###### Macro [BUILD\_MN][]([CHECK] [PTR] [MULTI] mninfo mnname) _# matrixnet_ <a name="macro_BUILD_MN"></a>
Generate MatrixNet data and access code using single command.
Alternative macro BUILD\_MNS() works faster and better for large files.

###### Macro [BUILD\_MNS][]([CHECK] NAME listname mninfos...) _# matrixnet_ <a name="macro_BUILD_MNS"></a>
Generate MatrixNet data and access code using separate commands for support code, interface and data.
Faster version of BUILD\_MN() macro for large files.

###### Macro [BUILD\_ONLY\_IF][](variables)  _# builtin_ <a name="macro_BUILD_ONLY_IF"></a>
Print warning if all variables are false. For example, BUILD\_ONLY\_IF(LINUX WIN32)

###### Macro [BUILD\_PLNS][](Src...) <a name="macro_BUILD_PLNS"></a>
Generate interface header plnmodels.h for Relev model (PLN)
Relev specific macro

###### Macro [BUILTIN\_PYTHON][](script\_path args... [CWD dir] [ENV key=value...] [TOOL tools...] [IN inputs...] [OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output]) [OUTPUT\_INCLUDES output\_includes...]) <a name="macro_BUILTIN_PYTHON"></a>
Run a python script with ymake python.
These macros are similar: RUN\_PROGRAM, LUA, PYTHON, BUILTIN\_PYTHON.

Parameters:
- script\_path - Path to the script.
- args... - Program arguments. Relative paths listed in TOOL, IN, OUT, STDOUT become absolute.
- CWD dir - Absolute path of the working directory.
- ENV key=value... - Environment variables.
- TOOL tools... - Auxiliary tool directories.
- IN inputs... - Input files.
- OUT[\_NOAUTO] outputs... - Output files. NOAUTO outputs are not automatically added to the build process.
- STDOUT[\_NOAUTO] output - Redirect the standard output to the output file.
- OUTPUT\_INCLUDES output\_includes... - Includes of the output files that are needed to build them.

For absolute paths use ${ARCADIA\_ROOT} and ${ARCADIA\_BUILD\_ROOT}, or
${CURDIR} and ${BINDIR} which are expanded where the outputs are used.

###### Macro [BUNDLE][](<Dir [NAME Name]>...) <a name="macro_BUNDLE"></a>
Brings build artefact from module Dir under optional Name to the current module (e.g. UNION)
If NAME is not specified, the name of the Dir's build artefact will be preserved
It makes little sense to specify BUNDLE on non-final targets and so this may stop working without prior notice.
Bundle on multimodule will select final target among multimodule variants and will fail if there are none or more than one.

###### Macro $[CFG\_VARS][] _# internal_ <a name="macro_CFG_VARS"></a>
Mark commands that embed Configuration variables into files

###### Macro [CFLAGS][]([GLOBAL compiler\_flag]\* compiler\_flags) <a name="macro_CFLAGS"></a>
Add the specified flags to the compilation command of C and C++ files.
@params: GLOBAL - Propagates these flags to dependent projects
Note: remember about the incompatibility flags for gcc and cl (to set flags specifically for cl.exe use MSVC\_FLAGS).

###### Macro [CGO\_CFLAGS][](Flags...) <a name="macro_CGO_CFLAGS"></a>
Compiler flags specific to CGO compilation

###### Macro [CGO\_LDFLAGS][](Files...) <a name="macro_CGO_LDFLAGS"></a>
Linker flags specific to CGO linking

###### Macro [CGO\_SRCS][](Files...) <a name="macro_CGO_SRCS"></a>
.go sources to be built with CGO

###### Macro [CHECK\_CONFIG\_H][](<conf\_header>) _# internal_ <a name="macro_CHECK_CONFIG_H"></a>
This internal macro adds checking code for configutaion header in external (contrib) library
The check is needed to avoid conflicts on certain types and functions available in arcadia
See https://a.yandex-team.ru/arc/trunk/arcadia/build/scripts/check\_config\_h.py for exact details

###### Macro [CHECK\_DEPENDENT\_DIRS][](...)  _#bultin_ <a name="macro_CHECK_DEPENDENT_DIRS"></a>
Ignored

###### Macro [CHECK\_JAVA\_DEPS][](<yes|no>) <a name="macro_CHECK_JAVA_DEPS"></a>
Check for different classes with duplicate name in classpath
Documentation: https://wiki.yandex-team.ru/yatool/test/

###### Macro [CLANG\_EMIT\_AST\_CXX][](Input Output Opts...) <a name="macro_CLANG_EMIT_AST_CXX"></a>
Emit Clang AST from .cpp file. CXXFLAGS and LLVM\_OPTS are passed in, while CGFLAGS and C\_FLAGS\_PLATFORM are not.
Note: Output name is used as is, no extension added

###### Macro [COMPILE\_C\_AS\_CXX][]() <a name="macro_COMPILE_C_AS_CXX"></a>
Not documented yet.

###### Macro [COMPILE\_SWIFT\_MODULE][](SRCS{input}[], BRIDGE\_HEADER{input}="", Flags...) <a name="macro_COMPILE_SWIFT_MODULE"></a>
Not documented yet.

###### Macro [CONDITIONAL\_PEERDIR][] <a name="macro_CONDITIONAL_PEERDIR"></a>
Not documented yet.

###### Macro [CONFIGURE\_FILE][](from to @ONLY) <a name="macro_CONFIGURE_FILE"></a>
Copy files with the replacement @CMakeListsVar@ variable CMakeListsVar from ya.make during the build.
Similar to the processing of files with the extension .in

###### Macro [CONLYFLAGS][]([GLOBAL compiler\_flag]\* compiler\_flags) <a name="macro_CONLYFLAGS"></a>
Add the specified flags to the compilation command of .c (but not .cpp) files.
@params: GLOBAL - Distributes these flags on dependent projects

###### Macro [COPY][] <a name="macro_COPY"></a>
Not documented yet.

###### Macro [COPY\_FILE][](File Destination) <a name="macro_COPY_FILE"></a>
Copy file to build root. It is possible to change both location and the name.
The file will be just copied. You should explicitly mention it in SRCS under new name for further processing

###### Macro [COPY\_FILE\_WITH\_DEPS][](File Destination [OUTPUT\_INCLUDES Deps...]) <a name="macro_COPY_FILE_WITH_DEPS"></a>
Copy file to build root and specify additional dependencies for it.
It is possible to change both location and the name.
The file will be just copied. You should explicitly mention it in SRCS under new name for further compilation.
File will be recompiled (but not re-copied) if any of specified dependencies changes

###### Macro [CREATE\_BUILDINFO\_FOR][](GenHdr) <a name="macro_CREATE_BUILDINFO_FOR"></a>
Creates header file to access some information about build specified via configuration variables
Unlike CREATE\_SVNVERSION\_FOR() it doesn't take revion information from VCS, it uses revision and SandboxTaskId passed via -D options to ya make

###### Macro [CREATE\_GO\_SVNVERSION\_FOR][]() <a name="macro_CREATE_GO_SVNVERSION_FOR"></a>
Creates a .go-file with information about the svn revision and various other things

Can lead to frequent rebuilding programs.
Partly off by setting to yes NO\_SVN\_DEPENDS

###### Macro [CREATE\_INIT\_PY][] <a name="macro_CREATE_INIT_PY"></a>
Not documented yet.

###### Macro [CREATE\_INIT\_PY\_FOR][] <a name="macro_CREATE_INIT_PY_FOR"></a>
Not documented yet.

###### Macro [CREATE\_INIT\_PY\_STRUCTURE][] <a name="macro_CREATE_INIT_PY_STRUCTURE"></a>
Not documented yet.

###### Macro [CREATE\_JAVA\_SVNVERSION\_FOR][]() <a name="macro_CREATE_JAVA_SVNVERSION_FOR"></a>
Creates a .java-file with information about the svn revision and various other things

Can lead to frequent rebuilding programs.
Partly off by setting to yes NO\_SVN\_DEPENDS

###### Macro [CREATE\_SVNVERSION\_FOR][]() <a name="macro_CREATE_SVNVERSION_FOR"></a>
Creates a .h-file with information about the svn revision and various other things

Can lead to frequent rebuilding programs.
Partly off by setting to yes NO\_SVN\_DEPENDS

###### Macro [CTEMPLATE\_VARNAMES][](File) <a name="macro_CTEMPLATE_VARNAMES"></a>
Generate File.varnames.h using contrib/libs/ctemplate/make\_tpl\_varnames\_h
Documentation: https://a.yandex-team.ru/arc/trunk/arcadia/contrib/libs/ctemplate/README.md

###### Macro [CUDA\_NVCC\_FLAGS][](compiler flags) <a name="macro_CUDA_NVCC_FLAGS"></a>
Add the specified flags to the compile line .cu-files.

###### Macro [CXXFLAGS][](compiler\_flags) <a name="macro_CXXFLAGS"></a>
Add the specified flags to the compilation command of .cpp (but not .c) files.

###### Macro [DARWIN\_SIGNED\_RESOURCE][](Resource, Relpath) <a name="macro_DARWIN_SIGNED_RESOURCE"></a>
Not documented yet.

###### Macro [DARWIN\_STRINGS\_RESOURCE][](Resource, Relpath) <a name="macro_DARWIN_STRINGS_RESOURCE"></a>
Not documented yet.

###### Macro [DATA][]([path...]) <a name="macro_DATA"></a>
Specifies the path to the data necessary test.
Valid: arcadia/<path> , arcadia\_tests\_data/<path>.

You can also specify the number of the resource required test in cbr format://<eng-id> (resource will be brought to the working directory of the test before it is started)
Used only inside TEST modules

Documentation: https://wiki.yandex-team.ru/yatool/test/#dannyeizrepozitorija

###### Macro [DEB\_VERSION][](File) <a name="macro_DEB_VERSION"></a>
Creates a header file DebianVersion.h define the DEBIAN\_VERSION taken from the File

###### Macro [DECIMAL\_MD5\_LOWER\_32\_BITS][](<fileName> [FUNCNAME funcName] [inputs...]) <a name="macro_DECIMAL_MD5_LOWER_32_BITS"></a>
Generates .cpp file <fileName> with one defined function 'const char\* <funcName>() { return "<calculated\_md5\_hash>"; }'.
<calculated\_md5\_hash> will be md5 hash for all inputs passed to this macro.

###### Macro [DECLARE\_EXTERNAL\_HOST\_RESOURCES\_BUNDLE][](name sbr:id platform name1 sbr:id1 platform1...)  _#builtin_ <a name="macro_DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE"></a>
Associate name with sbr-id on platform.

Ask devtools@yandex-team.ru if you need more information

###### Macro [DECLARE\_EXTERNAL\_RESOURCE][](name sbr:id name1 sbr:id1...)  _#builtin_ <a name="macro_DECLARE_EXTERNAL_RESOURCE"></a>
Associate name with sbr-id.

Ask devtools@yandex-team.ru if you need more information

###### Macro [DEFAULT][](varname value)  _#builtin_ <a name="macro_DEFAULT"></a>
Sets varname to value if value is not set yet

###### Macro [DEPENDENCY\_MANAGEMENT][](path/to/lib1 path/to/lib2 ...) <a name="macro_DEPENDENCY_MANAGEMENT"></a>
It allow you to lock the versions of the libraries from the contrib/java in one place,
  and all modules of the project when writing PEERDIR not specify a version.
For example, if the module says PEERDIR (contrib/java/junit/junit), and
  1. written DEPENDENCY\_MANAGEMENT(contrib/java/junit/junit/4.12),
     the PEERDIR is automatically replaced by contrib/java/junit/junit/4.12
  2. not written appropriate DEPENDENCY\_MANAGEMENT, PEERDIR automatically replaced
     with the default from contrib/java/junit/junit/ya.make.
     These defaults are always there and are supported in maven-import, which is put
     there the maximum(available in contrib/java ) version.
The property is transitive. That is, if module a depends on module B,
and PEERDIR is written in B(contrib/java/junit/junit), and this junit was replaced by junit-4.12, then junit-4.12 will come to A through B.
Allows you to lock the versions of the libraries from the contrib/java falling in the classpath transitive.
For example, if a module written DEPENDENCY\_MANAGEMENT(contrib/java/junit/junit/4.12),
but not written appropriate PEERDIR(contrib/java/junit/junit),
  1. if you are attracted by dependencies, such as contrib/java/junit/junit/4.11, it will be replaced by contrib/java/junit / junit / 4.12
  2. if the dependency does not extend any junit DEPENDENCY\_MANAGEMENT this has no effect
This property is not passed transitively. That is, if module a depends on B,
and B replaced itself transitively arriving junit-4.11 on junit-4.12, in a through B all the same will arrive junit-4.11.

In the case when the module is written at the same time DEPENDENCY\_MANAGEMENT(contrib/java/junit/junit/4.12) and PERDIR(contrib/java/junit/junit/4.11) PEERDIR wins.

###### Macro [DEPENDS][](path1 [path2...]) <a name="macro_DEPENDS"></a>
Buildable targets that should be brought to the test run.
It is reasonable to specify only final targets (like programs, DLLs or packages here). And only such targets are taken from multimodules
DEPENDS to UNION is the only exception from the rule above: UNIONs are transitively closed at DEPENDS bringing all dependencies to the test

###### Macro [DISABLE][](varname)  _#builtin_ <a name="macro_DISABLE"></a>
Sets varname to 'false'

###### Macro [DLL\_FOR][](path/to/lib [libname] [major\_ver [minor\_ver]] [EXPORTS symlist\_file])  _#builtin_ <a name="macro_DLL_FOR"></a>
Not documented yet.

###### Macro [DOCS\_CONFIG][](path) <a name="macro_DOCS_CONFIG"></a>
Specify path to config .yml file for DOCS multimodule if it differs from "%%project\_directory%%/mkdocs.yml".
Path must be either Arcadia root relative or lie in project directory.
@see: [DOCS](#multimodule\_DOCS)

###### Macro [DOCS\_DIR][](path) <a name="macro_DOCS_DIR"></a>
Specify directory with source .md files for DOCS multimodule if it differs from project directory.
Path must be Arcadia root relative.
@see: [DOCS](#multimodule\_DOCS)

###### Macro [DOCS\_VARS][](variable1=value1 variable2=value2 ...) <a name="macro_DOCS_VARS"></a>
Specify a set of default values of template variables for DOCS multimodule.
There must be no spaces around "=". Values will be treated as strings.
@see: [DOCS](#multimodule\_DOCS)

###### Macro [DUMPERF\_CODEGEN][](Prefix) <a name="macro_DUMPERF_CODEGEN"></a>
A special case BASE\_CODEGEN, in which the extsearch/images/robot/tools/dumperf/codegen tool is used

###### Macro IF(expression) .. [ELSE][]IF(expression) .. ELSE() .. ENDIF()  _#builtin_ <a name="macro_ELSE"></a>
Not documented yet.

###### Macro IF(expression) .. [ELSEIF][](expression) .. ELSE() .. ENDIF()  _#builtin_ <a name="macro_ELSEIF"></a>
Not documented yet.

###### Macro [ENABLE][](varname)  _#builtin_ <a name="macro_ENABLE"></a>
Sets varname to 'true'

###### Macro [END][]()  _# builtin_ <a name="macro_END"></a>
The end of the module

###### Macro IF(expression) .. ELSEIF(expression) .. ELSE() .. [ENDIF][]()  _#builtin_ <a name="macro_ENDIF"></a>
Not documented yet.

###### Macro [ENV][](key[=value]) <a name="macro_ENV"></a>
Sets env variable key to value (gets value from system env by default)

###### Macro [EXCLUDE][] <a name="macro_EXCLUDE"></a>
EXCLUDE(prefixes)
The macro is in development.
Specifies which libraries should be excluded from the classpath.

###### Macro [EXCLUDE\_TAGS][](tags...)  _# builtin_ <a name="macro_EXCLUDE_TAGS"></a>
Instantiate from multimodule all variants except ones with tags listed

###### Macro [EXPORTS\_SCRIPT][](symlist\_file) <a name="macro_EXPORTS_SCRIPT"></a>
This allows to specify exports script using separate macro within DLL and DLL-derived modules
The other option use EXPORTS parameter of the module itself
@see: [DLL](#module\_DLL)

###### Macro [EXPORT\_MAPKIT\_PROTO][]() _# internal deprecated_ <a name="macro_EXPORT_MAPKIT_PROTO"></a>
This macro is a temporary one and should be changed to EXPORT\_YMAPS\_PROTO
when transition of mapsmobi to arcadia is finished

###### Macro [EXPORT\_YMAPS\_PROTO][]() _# maps-specific_ <a name="macro_EXPORT_YMAPS_PROTO"></a>
Maps-specific .proto handling: IMPORT\_YMAPS\_PROTO() + maps protobuf namespace

###### Macro [EXTERNAL\_JAR][](library.jar) <a name="macro_EXTERNAL_JAR"></a>
Is used to indicate the collected external java libraries inside the project JAVA\_LIBRARY()

Documentation: https://wiki.yandex-team.ru/yatool/java/#ispolzovanievneshnixmavenbibliotek

###### Macro [EXTERNAL\_RESOURCE][](...)  _#builtin deprecated_ <a name="macro_EXTERNAL_RESOURCE"></a>
Deprecated

###### Macro [EXTRADIR][](...)  _#bultin_ <a name="macro_EXTRADIR"></a>
Ignored

###### Macro [EXTRALIBS][](liblist)  _# builtin_ <a name="macro_EXTRALIBS"></a>
Add external dynamic libraries during program linkage stage

###### Macro [EXTRALIBS\_STATIC][](Libs...) <a name="macro_EXTRALIBS_STATIC"></a>
Add the specified external static libraries to the program link

###### Macro [FAT\_RESOURCE][] <a name="macro_FAT_RESOURCE"></a>
Not documented yet.

###### Macro [FILES][] <a name="macro_FILES"></a>
Not documented yet.

###### Macro [FLAT\_JOIN\_SRCS\_GLOBAL][](Out Src...) <a name="macro_FLAT_JOIN_SRCS_GLOBAL"></a>
Join set of sources into single file named Out and send it for further processing as if it were listed as SRCS(GLOBAL Out)
This macro places all files into single file, so will work with any sources.
You should specify file name with the extension as Out

###### Macro [FORK\_SUBTESTS][]() <a name="macro_FORK_SUBTESTS"></a>
Splits the test run in chunks on subtests.
The number of chunks can be overridden using the macro SPLIT\_FACTOR.

Allows to run tests in parallel. Supported C++ ut and PyTest.

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [FORK\_TESTS][]() <a name="macro_FORK_TESTS"></a>
Splits a test run on chunks classes.
The number of chunks can be overridden using the macro SPLIT\_FACTOR.

Allows to run tests in parallel. Supported C++ ut and PyTest.

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [FORK\_TEST\_FILES][]() <a name="macro_FORK_TEST_FILES"></a>
Only for PYTEST: splits a file executable with the tests on chunks in the files listed in TEST\_SRCS
Compatible with FORK\_(SUB)TESTS.

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [FROM\_MDS][]([FILE] key OUT\_[NOAUTO] <files from resource>) <a name="macro_FROM_MDS"></a>
Downloads the file from MDS using key, unpacks (if not explicitly specified word FILE) and adds the files with the specified names in the build.
If the files do not compile, then you need to use the parameter OUT\_NOAUTO.
Specify the directory, as it build models - input and output data for the node assemblies must be determined.

###### Macro [FROM\_SANDBOX][]([FILE] resource\_id is OUT\_[NOAUTO] <files from resource>) <a name="macro_FROM_SANDBOX"></a>
Downloads the file from the sandbox to the resource number, unpacks (if not explicitly specified word FILE) and adds the files with the specified names in the build.
If the files do not compile, then you need to use the parameter OUT\_NOAUTO.
Specify the directory, as it build models - input and output data for the node assemblies must be determined.

###### Macro FUZZ\_DICT(path1 [path2...]) <a name="macro_FUZZ_DICTS"></a>
Allows you to specify dictionaries, relative to the root of Arcadia, which will be used in Fuzzing.
Libfuzzer and AFL use a single syntax for dictionary descriptions.

Should only be used in FUZZ modules
Documentation: https://wiki.yandex-team.ru/yatool/fuzzing/

###### Macro [FUZZ\_OPTS][](opt1 [Opt2...]) <a name="macro_FUZZ_OPTS"></a>
 Overrides or adds options to the corpus mining and fuzzer run.
 Currently supported only Libfuzzer, so you should use the options for it.
 Example:
 Fuzz\_opts (
-Max\_len = 1024
-RSS\_LIMIT\_MB = 8192
 )

 Should only be used in FUZZ modules
 Documentation: https://wiki.yandex-team.ru/yatool/fuzzing/

###### Macro [GENERATED\_SRCS][](srcs... PARSE\_META\_FROM cpp\_srcs... [OUTPUT\_INCLUDES output\_includes...] [OPTIONS]) <a name="macro_GENERATED_SRCS"></a>
srcs... - list of text files which will be generated during build time by templates. Each template must be
    placed to the place in source tree where corresponding source file should be generated. Name of
    template must be "<name\_of\_src\_file>.template". For example if you want to generate file "example.cpp"
    then template should be named "example.cpp.template".
PARSE\_META\_FROM cpp\_srcs... - list of C++ source files (.cpp, .h) which will be parsed using clang library
    and metainformation extracted from the files will be made available for templates. Example of
    template code fragment using metainformation: {{ meta.objects["@N@std@S@string"].name }}
OUTPUT\_INCLUDES output\_includes... - in cases when build system parser fails to determine all headers
    which generated files include, you can specify additional headers here. In a normal situation this should
    not be needed and this feature could be removed in the future.
OPTIONS - additional options for code\_generator utility

Jinja 2 template engine is used for template generation.
Examples of templates can be found in directory market/tools/code\_generator/templates.
Metainformation does not contain entries for every object declared in C++ files specified in PARSE\_META\_FROM
parameter. To include some object into consideration you need to mark it by attribute. Attributes can
automatically add more attributes to dependent objects. This behavior depends on attribute definition.

More information will be available (eventually:) here: https://wiki.yandex-team.ru/Users/denisk/codegenerator/

###### Macro [GENERATE\_ENUM\_SERIALIZATION][](File.h) <a name="macro_GENERATE_ENUM_SERIALIZATION"></a>
Create serialization support for enumeration members defined in the header (String <-> Enum conversions) and compile it into the module.
Documentation: https://wiki.yandex-team.ru/yatool/HowToWriteYaMakeFiles/

###### Macro [GENERATE\_ENUM\_SERIALIZATION\_WITH\_HEADER][](File.h) <a name="macro_GENERATE_ENUM_SERIALIZATION_WITH_HEADER"></a>
Create serialization support for enumeration members defined in the header (String <-> Enum conversions) and compile it into the module
Provide access to serialization functions via generated header File\_serialized.h
Documentation: https://wiki.yandex-team.ru/yatool/HowToWriteYaMakeFiles/

###### Macro [GENERATE\_PY\_PROTOS][](ProtoFiles...) _# deprecated_ <a name="macro_GENERATE_PY_PROTOS"></a>
Generate python bindings for protobuf files.
Macro is obsolete and not recommended for use!

###### Macro [GENERATE\_SCRIPT][] <a name="macro_GENERATE_SCRIPT"></a>
heretic@ promised to make tutorial here
Don't forget
Feel free to remind

###### Macro [GEN\_SCHEEME2][](scheeme\_name from\_file dependent\_files...) <a name="macro_GEN_SCHEEME2"></a>
Generates a C++ description for structure(contains the field RecordSig) in the specified file (and connected).

1. ${scheeme\_name}.inc - the name of the generated file.
2. Use an environment variable - DATAWORK\_SCHEEME\_EXPORT\_FLAGS that allows to specify flags to tools/structparser

@example:

    SET(DATAWORK\_SCHEEME\_EXPORT\_FLAGS --final\_only -m "::")

all options are passed to structparser (in this example --final\_only - do not export heirs with public base that contains the required field,,- m "::" only from the root namespace)
sets in extra option

@example:

    SET(EXTRACT\_STRUCT\_INFO\_FLAGS -f \"const static ui32 RecordSig\"
        -u \"RecordSig\" -n${scheeme\_name}SchemeInfo ----gcc44\_no\_typename no\_complex\_overloaded\_func\_export
        ${DATAWORK\_SCHEEME\_EXPORT\_FLAGS})

for compatibility with C++ compiler and the external environment.
(the details necessary to look in tools/structparser)

###### Macro [GLOBAL\_SRCS][](filenames...) <a name="macro_GLOBAL_SRCS"></a>
Make all source files listed as GLOBAL
Call to GLOBAL\_SRCS macro is equivalent to call to SRCS macro when each source file is marked with GLOBAL keyword.
Arcadia root relative or project dir relative paths are supported for filenames arguments. GLOBAL keyword is not
recognized for GLOBAL\_SRCS in contrast to SRCS macro.

@example:
Consider the file to ya.make:
```
    LIBRARY()
    GLOBAL\_SRCS(foo.cpp bar.cpp)
    END()
```
@see: [SRCS()](#macro\_SRCS)

###### Macro [GO\_ASM\_FLAGS][](flags) <a name="macro_GO_ASM_FLAGS"></a>
Add the specified flags to the go asm compile icommand line.

###### Macro [GO\_CGO1\_FLAGS][](flags) <a name="macro_GO_CGO1_FLAGS"></a>
Add the specified flags to the go cgo compile command line.

###### Macro [GO\_CGO2\_FLAGS][](flags) <a name="macro_GO_CGO2_FLAGS"></a>
Add the specified flags to the go cgo compile command line.

###### Macro [GO\_COMPILE\_FLAGS][](flags) <a name="macro_GO_COMPILE_FLAGS"></a>
Add the specified flags to the go compile command line.

###### Macro [GO\_FAKE\_OUTPUT][](go-src-files...) <a name="macro_GO_FAKE_OUTPUT"></a>
Not documented yet.

###### Macro [GO\_LDFLAGS][](Flags...) <a name="macro_GO_LDFLAGS"></a>
Link flags for GO\_PROGRAM linking from .go sources

###### Macro [GO\_LINK\_FLAGS][](flags) <a name="macro_GO_LINK_FLAGS"></a>
Add the specified flags to the go link command line.

###### Macro [GO\_PACKAGE\_NAME][](Name) <a name="macro_GO_PACKAGE_NAME"></a>
Override name of a Go package.

###### Macro [GO\_SRCS][](Files...) _# internal_ <a name="macro_GO_SRCS"></a>
This macro shouldn't be used in ya.make files, use SRCS() instead.
This is internal macro collecting .go sources for processing within Go modules (GO\_PROGRAM and GO\_LIBRARY)

###### Macro [GO\_TEST\_FOR][](path/to/module)  _#builtin_ <a name="macro_GO_TEST_FOR"></a>
Produces go test for specified module

###### Macro [GO\_TEST\_SRCS][](Files...) <a name="macro_GO_TEST_SRCS"></a>
.go sources for internal tests of a module

###### Macro [GO\_XTEST\_SRCS][](Files...) <a name="macro_GO_XTEST_SRCS"></a>
.go sources for external tests of a module

###### Macro [GRPC][]() <a name="macro_GRPC"></a>
Emit GRPC code for all .proto files in a PROTO\_LIBRARY
This works for all available PROTO\_LIBRARY versions (C++, Python 2.x, Pyton 3.x, Java and Go)

###### Macro IDEA\_EXCLUDE\_RESOURCE\_DIRS(<excluded dirs>) <a name="macro_IDEA_EXCLUDE_DIRS"></a>
Macro excludes specified resource directories from an idea project generated by ya ide idea

###### Macro [IDEA\_RESOURCE\_DIRS][](<additional dirs>) <a name="macro_IDEA_RESOURCE_DIRS"></a>
Macro sets specified resource directories in an idea project generated by ya ide idea

###### Macro [IF][](expression) .. ELSEIF(expression) .. ELSE() .. ENDIF()  _#builtin_ <a name="macro_IF"></a>
Not documented yet.

###### Macro [IMPORT\_YMAPS\_PROTO][]() _# maps-specific_ <a name="macro_IMPORT_YMAPS_PROTO"></a>
Maps-specific .proto handling: base proto dicrectory setup

###### Macro [INCLUDE][](filename)  _#builtin_ <a name="macro_INCLUDE"></a>
Not documented yet.

###### Macro [INCLUDE\_TAGS][](tags...)  _# builtin_ <a name="macro_INCLUDE_TAGS"></a>
Additionally instantiate from multimodule all variants with tags listed (overrides default)

###### Macro [INDUCED\_DEPS][](extension path)  _#builtin_ <a name="macro_INDUCED_DEPS"></a>
Permits one to specify include/import path as a generated path. path is considered as output by tools and system will rebuild files on demand

###### Macro [IOS\_APP\_ASSETS\_FLAGS][](Flags...) <a name="macro_IOS_APP_ASSETS_FLAGS"></a>
Not documented yet.

###### Macro [IOS\_APP\_COMMON\_FLAGS][](Flags...) <a name="macro_IOS_APP_COMMON_FLAGS"></a>
Not documented yet.

###### Macro [IOS\_APP\_SETTINGS][] <a name="macro_IOS_APP_SETTINGS"></a>
Not documented yet.

###### Macro [IOS\_ASSETS][] <a name="macro_IOS_ASSETS"></a>
Not documented yet.

###### Macro [JAVAC\_FLAGS][](Args...) <a name="macro_JAVAC_FLAGS"></a>
Set additional Java compilation flags

###### Macro [JAVA\_IGNORE\_CLASSPATH\_CLASH\_FOR][]([classes]) <a name="macro_JAVA_IGNORE_CLASSPATH_CLASH_FOR"></a>
Ignore classpath clash test fails for classes

###### Macro [JAVA\_MODULE][] <a name="macro_JAVA_MODULE"></a>
Not documented yet.

###### Macro [JAVA\_SRCS][](srcs) <a name="macro_JAVA_SRCS"></a>
Using the macro JAVA\_SRCS() specifies the java source files and resources. A macro can be contained in any of four java modules.
Keywords:
1. X SRCDIR - specify the directory x is performed relatively to search the source code for these patterns. If there is no SRCDIR, the source will be searched relative to the module directory.
2. PACKAGE\_PREFIX x - use if source paths relative to the SRCDIR does not coincide with the full class names. For example, if all sources of module are in the same package, you can create a directory package/name , and just put the source code in the SRCDIR and specify PACKAGE\_PREFIX package.name.

@example:
 - example/ya.make

    JAVA\_PROGRAM()
    JAVA\_SRCS(SRCDIR src/main/java \*\*/\*)
    END()

 - example/src/main/java/ru/yandex/example/HelloWorld.java

    package ru.yandex.example;
    public class HelloWorld {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }

    macro JAVA\_SRCS(Args...) {
        SET\_APPEND(JAVA\_SRCS\_VALUE $ARGS\_DELIM $Args)
    }
Documentation: https://wiki.yandex-team.ru/yatool/java/#javasrcs

###### Macro [JAVA\_TEST][] <a name="macro_JAVA_TEST"></a>
Not documented yet.

###### Macro [JAVA\_TEST\_DEPS][] <a name="macro_JAVA_TEST_DEPS"></a>
Not documented yet.

###### Macro [JOINSRC][]() <a name="macro_JOINSRC"></a>
Not documented yet.

###### Macro [JOIN\_SRCS][](Out Src...) <a name="macro_JOIN_SRCS"></a>
Join set of sources into single file named Out and send it for further processing.
This macro doesn't place all file into Out, it emits #include<Src>... Use the for C++ source files only.
You should specify file name with the extension as Out

###### Macro [JOIN\_SRCS\_GLOBAL][](Out Src...) <a name="macro_JOIN_SRCS_GLOBAL"></a>
Join set of sources into single file named Out and send it for further processing as if it were listed as SRCS(GLOBAL Out)
This macro doesn't place all file into Out, it emits #include<Src>... Use the for C++ source files only.
You should specify file name with the extension as Out

###### Macro [JVM\_ARGS][](Args...) <a name="macro_JVM_ARGS"></a>
Arguments to run Java programs in tests
Documentation: https://wiki.yandex-team.ru/yatool/test/

###### Macro [LDFLAGS][](LinkerFlags...) <a name="macro_LDFLAGS"></a>
Add flags to the linkage command line of executable or shared library/dll.
Note: LDFLAGS are always global - when set in the library module they will affect all programs/dlls/tests the library is linked into.
Note: remember about the incompatibility of flags for gcc and cl.

###### Macro [LDFLAGS\_FIXED][](LinkerFlags...) <a name="macro_LDFLAGS_FIXED"></a>
Fixed version of LDFLAGS "macro" which does not substitute build variables in some cases.
Use this fixed version to workaround.
Note: LDFLAGS are always global - when set in the library module they will affect all programs/dlls/tests the library is linked into.
Note: remember about the incompatibility of flags for gcc and cl.

###### Macro [LICENSE][](licenses...) <a name="macro_LICENSE"></a>
Specify the license of the module, separated by spaces

A license must be prescribed for contribs

###### Macro [LINT][](<none|base|strict>) <a name="macro_LINT"></a>
Set linting levem for sources of the module

###### Macro [LJ\_21\_ARCHIVE][](NAME Name LuaFiles...) _# deprecated_ <a name="macro_LJ_21_ARCHIVE"></a>
Precompile .lua files using LuaJIT 2.1 and archive both sources and results using sources names as keys

###### Macro [LJ\_ARCHIVE][](NAME Name LuaFiles...) <a name="macro_LJ_ARCHIVE"></a>
Precompile .lua files using LuaJIT and archive both sources and results using sources names as keys

###### Macro [LLVM\_BC][] <a name="macro_LLVM_BC"></a>
Not documented yet.

###### Macro [LLVM\_COMPILE\_C][](Input Output Opts...) <a name="macro_LLVM_COMPILE_C"></a>
Emit LLVM bytecode from .c file. BC\_CFLAGS, LLVM\_OPTS and C\_FLAGS\_PLATFORM are passed in, while CGFLAGS are not.
Note: Output name is used as is, no extension added

###### Macro [LLVM\_COMPILE\_CXX][](Input Output Opts...) <a name="macro_LLVM_COMPILE_CXX"></a>
Emit LLVM bytecode from .cpp file. BC\_CXXFLAGS, LLVM\_OPTS and C\_FLAGS\_PLATFORM are passed in, while CGFLAGS are not.
Note: Output name is used as is, no extension added

###### Macro [LLVM\_COMPILE\_LL][](Input Output Opts...) <a name="macro_LLVM_COMPILE_LL"></a>
Compile LLVM bytecode to object representation
Note: Output name is used as is, no extension added

###### Macro [LLVM\_LINK][](Output Inputs...) <a name="macro_LLVM_LINK"></a>
Call llvm-link on set of Inputs to produce Output.
Note: Unlike many other macros output argument goes first. Output name is used as is, no extension added.

###### Macro [LLVM\_OPT][](Input Output Opts...) <a name="macro_LLVM_OPT"></a>
Call llvm-opt with set of Opts on Input to produce Output
Note: Output name is used as is, no extension added.

###### Macro [LSAN\_SUPPRESSIONS][] <a name="macro_LSAN_SUPPRESSIONS"></a>
LSAN\_SUPPRESSIONS() - allows to specify files with suppression notation which will be used by default.
See https://clang.llvm.org/docs/AddressSanitizer.html#suppressing-memory-leaks
for details.

###### Macro [LUA][](script\_path args... [CWD dir] [ENV key=value...] [TOOL tools...] [IN inputs...] [OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output] [OUTPUT\_INCLUDES output\_includes...]) <a name="macro_LUA"></a>
Run a lua script.
These macros are similar: RUN\_PROGRAM, LUA, PYTHON, BUILTIN\_PYTHON.

Parameters:
- script\_path - Path to the script.
- args... - Program arguments. Relative paths listed in TOOL, IN, OUT, STDOUT become absolute.
- CWD dir - Absolute path of the working directory.
- ENV key=value... - Environment variables.
- TOOL tools... - Auxiliary tool directories.
- IN inputs... - Input files.
- OUT[\_NOAUTO] outputs... - Output files. NOAUTO outputs are not automatically added to the build process.
- STDOUT[\_NOAUTO] output - Redirect the standard output to the output file.
- OUTPUT\_INCLUDES output\_includes... - Includes of the output files that are needed to build them.

For absolute paths use ${ARCADIA\_ROOT} and ${ARCADIA\_BUILD\_ROOT}, or
${CURDIR} and ${BINDIR} which are expanded where the outputs are used.

###### Macro [MACROS\_WITH\_ERROR][] <a name="macro_MACROS_WITH_ERROR"></a>
Not documented yet.

###### Macro [MAPKITIDL][](idl-file-name... <a name="macro_MAPKITIDL"></a>
[OUT\_DIR output-dir]
[IDL\_INCLUDES idl-dirs...]
[FILTER filters...])
[FILTER\_OUT filters...])
[GLOBAL\_OUTPUTS]
Generate bindings to target platform language.
(Used for mobile MapKit project)
1. idl-file-name... - a list of \*.idl files to process
2. output-dir - a base root of output directory
3. idl-dirs - a list of directories where to search for imported \*.idl files
4. filters - a list of extentions used to filter outputs and output includes

###### Macro [MAPKIT\_ADDINCL][](Dirs...) <a name="macro_MAPKIT_ADDINCL"></a>
Not documented yet.

###### Macro [MAPKIT\_ENABLE\_WHOLE\_ARCHIVE][]() _# internal deprecated_ <a name="macro_MAPKIT_ENABLE_WHOLE_ARCHIVE"></a>
This macro is a temporary one. Its use is allowed in maps/mapsmobi dir only.

###### Macro [MAPSMOBI\_SRCS][](filenames...) <a name="macro_MAPSMOBI_SRCS"></a>
Make all source files listed as GLOBAL or not (depending on the value of
MAPSMOBI\_USE\_SRCS\_GLOBAL). Be careful since the value of
MAPSMOBI\_USE\_SRCS\_GLOBAL matters! If the value of this variable is equal to
GLOBAL then call to MAPSMOBI\_SRCS() macro behaves like call to
GLOBAL\_SRCS() macro otherwise the value of MAPSMOBI\_USE\_SRCS\_GLOBAL is
treated as a file name and a call to MAPSMOBI\_SRCS() macro behaves like a
call to SRCS() macro with additional iargument which is the value of
MAPSMOBI\_USE\_SRCS\_GLOBAL variable

###### Macro [MASMFLAGS][](compiler flags) <a name="macro_MASMFLAGS"></a>
Add the specified flags to the compilation command of .masm files.

###### Macro [MAVEN\_GROUP\_ID][](group\_id\_for\_maven\_export) <a name="macro_MAVEN_GROUP_ID"></a>
Set maven export group id for JAVA\_PROGRAM() and JAVA\_LIBRARY()

###### Macro [MESSAGE][]([severity] message)  _# builtin_ <a name="macro_MESSAGE"></a>
Print message with given severity level (STATUS, FATAL\_ERROR)

###### Macro [METAQUERYFILES][](filenames...) _#deprecated_ <a name="macro_METAQUERYFILES"></a>
Deprecated

###### Macro [MSVC\_FLAGS][]([GLOBAL compiler\_flag]\* compiler\_flags) <a name="macro_MSVC_FLAGS"></a>
Add the specified flags to the compilation line of C/C++files.
Flags apply only if the compiler used is MSVC (cl.exe)

###### Macro [MX\_FORMULAS][](BinFiles...) _# deprecated, matrixnet_ <a name="macro_MX_FORMULAS"></a>
Create MatrixNet formulas archive

###### Macro [NEED\_CHECK][]() <a name="macro_NEED_CHECK"></a>
Commits to the project marked with this macro will be blocked by pre-commit check and then will be
automatically merged to trunk only if there is no new broken build targets in check results.
The use of this macro is disabled by default.

###### Macro [NEED\_REVIEW][]() _# deprecated_ <a name="macro_NEED_REVIEW"></a>
Mark the project as needing review.
Reviewers are listed in the macro OWNER. The use of this macro is disabled by default.
Details can be found here: https://clubs.at.yandex-team.ru/arcadia/6104

###### Macro [NO\_BUILD\_IF][](variables)  _# builtin_ <a name="macro_NO_BUILD_IF"></a>
Print warning if some variable is true

###### Macro [NO\_CHECK\_IMPORTS][]([patterns]) <a name="macro_NO_CHECK_IMPORTS"></a>
Do not run checks on imports of Python modules.
Optional parameter mask patterns describes the names of the modules that do not need to check.

###### Macro NO\_SANITIZE() <a name="macro_NO_CLANG_COVERAGE"></a>
Disable heavyweight clang coverage for the module

###### Macro [NO\_CODENAVIGATION][]() _# incternal_ <a name="macro_NO_CODENAVIGATION"></a>
Disable codenaviagtion for a module. Needed to avoid PEERDIR loops in codenavigation support.
Most probably you'll never need this. If you think you need, please contact devtools@ for assistance.

###### Macro [NO\_COMPILER\_WARNINGS][]() <a name="macro_NO_COMPILER_WARNINGS"></a>
Diasable all compiler warnings in the module.
Priorities: NO\_COMPILER\_WARNINGS > NO\_WERROR > WERROR\_MODE > WERROR

###### Macro [NO\_DEBUG\_INFO][]() <a name="macro_NO_DEBUG_INFO"></a>
Not documented yet.

###### Macro [NO\_GPL][]() <a name="macro_NO_GPL"></a>
To check the license of the dependencies, indicated by macro LICENSE, and fail the build in case of detection of GPL-like licenses.
Do not use with LIBRARY().
Issues a warning if the detected dependencies from contrib-ov in which the license is not specified.

###### Macro [NO\_JOIN\_SRC][]() _# deprecated, does-nothing_ <a name="macro_NO_JOIN_SRC"></a>
This macro currently does nothing. This is default behavior which cannot be overriden at module level.

###### Macro [NO\_LIBC][]() <a name="macro_NO_LIBC"></a>
Exclude dependencies on C++ and C runtimes (including util, musl and libeatmydata)
Note: use this with care. libc most likely will be linked into executable anyway,
so using libc headers/functions may not be detected at build time and may lead to unpredictable behavors at configure time.

###### Macro [NO\_LINT][]() <a name="macro_NO_LINT"></a>
Do not check for style files included in PY\_SRCS, TEST\_SRCS, JAVA\_SRCS

###### Macro [NO\_NEED\_CHECK][]() <a name="macro_NO_NEED_CHECK"></a>
Commits to the project marked with this macro will not be affected by higher-level NEED\_CHECK macro.

###### Macro [NO\_OPTIMIZE][]() <a name="macro_NO_OPTIMIZE"></a>
Build code without any optimizations (-O0 mode)

###### Macro [NO\_OPTIMIZE\_PY\_PROTOS][]() _# internal_ <a name="macro_NO_OPTIMIZE_PY_PROTOS"></a>
Disable Python proto optimization using embedding corresponding C++ code into binary.
Python protobuf runtime will use C++ implementation instead of Python one if former is available
This is default mode only for some system libraries

###### Macro [NO\_PLATFORM][]() <a name="macro_NO_PLATFORM"></a>
Exclude dependencies on C++ and C runtimes (including util, musl and libeatmydata) and set NO\_PLATFORM variable for special processing.
Note: use this with care. libc most likely will be linked into executable anyway,
so using libc headers/functions may not be detected at build time and may lead to unpredictable behavors at configure time.

###### Macro [NO\_PLATFORM\_RESOURCES][]() _# incternal_ <a name="macro_NO_PLATFORM_RESOURCES"></a>
Exlude dependency on platform resources libraries.
Most probably you'll never need this. If you think you need, please contact devtools@ for assistance.

###### Macro [NO\_PYTHON\_INCLUDES][]() _# internal_ <a name="macro_NO_PYTHON_INCLUDES"></a>
Disable dependencies on libraries providing Python headers.
This is only used in Python libraries themselves to avoid PEERDIR loops

###### Macro [NO\_RUNTIME][]() <a name="macro_NO_RUNTIME"></a>
This macro:
1. Sets the ENABLE(NOUTIL) + DISABLE(USE\_INTERNAL\_STL)
2. If the project that contains the macro NO\_RUNTIME(), peerdir-it project does not contain NO\_RUNTIME() => Warning
Note: use this with care. Arcadia STL most likely will be linked into executable anyway,
so using STL headers/functions/classes may not be detected at build time and may lead to unpredictable behavors at configure time.

###### Macro [NO\_SANITIZE][]() <a name="macro_NO_SANITIZE"></a>
Disable all sanitizers for the module

###### Macro [NO\_SANITIZE\_COVERAGE][]() <a name="macro_NO_SANITIZE_COVERAGE"></a>
Disable lightweight coverage (-fsanitize-coverage) for the module

###### Macro [NO\_SSE4][]() <a name="macro_NO_SSE4"></a>
Compile module without SSE4

###### Macro [NO\_UTIL][]() <a name="macro_NO_UTIL"></a>
Build module without dependency on util.
Note: use this with care. Util most likely will be linked into executable anyway,
so using util headers/functions/classes may not be detected at build time and may lead to unpredictable behavors at configure time.

###### Macro [NO\_WERROR][]() <a name="macro_NO_WERROR"></a>
Override WERROR() behavior
Priorities: NO\_COMPILER\_WARNINGS > NO\_WERROR > WERROR\_MODE > WERROR

###### Macro [NO\_WSHADOW][]() <a name="macro_NO_WSHADOW"></a>
Disable C++ shadowing warnings

###### Macro [ONLY\_TAGS][](tags...)  _# builtin_ <a name="macro_ONLY_TAGS"></a>
Instantiate from multimodule only variants with tags listed

###### Macro [OPTIMIZE\_PY\_PROTOS][]()  _# internal_ <a name="macro_OPTIMIZE_PY_PROTOS"></a>
Enable Python proto optimization using embedding corresponding C++ code into binary.
Python protobuf runtime will use C++ implementation instead of Python one if former is available
This is default mode for most PROTO\_LIBRARY's and PY\_LIBRARY's, some system ones being exceptions

###### Macro [OWNER][](owners...)  _# builtin_ <a name="macro_OWNER"></a>
Add reviewers/responsibles of the code.
In the OWNER macro you can use:
1. login-s from staff.yandex-team.ru
2. Review group (to specify the Code-review group need to use the prefix g:)

Ask devtools@yandex-team.ru if you need more information

###### Macro [PACK][](archive\_type) <a name="macro_PACK"></a>
Is placed inside the PACKAGE module, packs the build results tree to the archive with specified extension.

###### Macro [PACKAGE\_STRICT][]() <a name="macro_PACKAGE_STRICT"></a>
Not documented yet.

###### Macro [PARTITIONED\_RECURSE][]([BALANCING\_CONFIG config] [LOCAL] dirs...)  _# builtin_ <a name="macro_PARTITIONED_RECURSE"></a>
Add directories to the build
All projects must be reachable from the root chain RECURSE() for monorepo continuous integration functionality.
Arguments are processed in chunks

###### Macro [PARTITIONED\_RECURSE\_FOR\_TESTS][]([BALANCING\_CONFIG config] [LOCAL] dirs...)  _# builtin_ <a name="macro_PARTITIONED_RECURSE_FOR_TESTS"></a>
Add directories to the build if tests are demanded.
Arguments are processed in chunks

###### Macro [PARTITIONED\_RECURSE\_ROOT\_RELATIVE][]([BALANCING\_CONFIG config] dirlist)  _# builtin_ <a name="macro_PARTITIONED_RECURSE_ROOT_RELATIVE"></a>
In comparison with RECURSE(), in dirlist there must be a directory relative to the root (${ARCADIA\_ROOT}).
Arguments are processed in chunks

###### Macro [PEERDIR][](dirs...)  _# builtin_ <a name="macro_PEERDIR"></a>
Specify project dependencies
Indicates that the project depends on all of the projects from the list of dirs.
Libraries from these directories will be collected and linked to the current target if the target is executable or sharedlib/dll.
If the current target is a static library, the specified directories will not be built, but they will be linked to any executable target that will link the current library.
@params:
1. As arguments PERDIR you can only use the LIBRARY directory (the directory with the PROGRAM/DLL and derived from them are prohibited to use as arguments PEERDIR).
2. ADDINCL Keyword ADDINCL (written before the specified directory), adds the flag -I<path to library> the flags to compile the source code of the current project.
Perhaps it may be removed in the future (in favor of a dedicated ADDINCL)

###### Macro [PIRE\_INLINE][](FILES...) <a name="macro_PIRE_INLINE"></a>
Not documented yet.

###### Macro [PIRE\_INLINE\_CMD][] <a name="macro_PIRE_INLINE_CMD"></a>
yweb specific

###### Macro [PRINT\_MODULE\_TYPE][] <a name="macro_PRINT_MODULE_TYPE"></a>
Not documented yet.

###### Macro [PROCESS\_DOCS][] <a name="macro_PROCESS_DOCS"></a>
Not documented yet.

###### Macro [PROCESS\_DOCSLIB][] <a name="macro_PROCESS_DOCSLIB"></a>
Not documented yet.

###### Macro [PROTO2FBS][](InputProto) <a name="macro_PROTO2FBS"></a>
Produce flatbuf schema out of protobuf descrition

###### Macro [PROVIDES][](Name...) <a name="macro_PROVIDES"></a>
Specifies provided features. The names must be C identifiers.
Different libraries providing the same features can not be linked into one program.

###### Macro [PYTHON][](script\_path args... [CWD dir] [ENV key=value...] [TOOL tools...] [IN inputs...] [OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output] [OUTPUT\_INCLUDES output\_includes...]) <a name="macro_PYTHON"></a>
Run a python script with contrib/tools/python.
These macros are similar: RUN\_PROGRAM, LUA, PYTHON, BUILTIN\_PYTHON.

Parameters:
- script\_path - Path to the script.
- args... - Program arguments. Relative paths listed in TOOL, IN, OUT, STDOUT become absolute.
- CWD dir - Absolute path of the working directory.
- ENV key=value... - Environment variables.
- TOOL tools... - Auxiliary tool directories.
- IN inputs... - Input files.
- OUT[\_NOAUTO] outputs... - Output files. NOAUTO outputs are not automatically added to the build process.
- STDOUT[\_NOAUTO] output - Redirect the standard output to the output file.
- OUTPUT\_INCLUDES output\_includes... - Includes of the output files that are needed to build them.

For absolute paths use ${ARCADIA\_ROOT} and ${ARCADIA\_BUILD\_ROOT}, or
${CURDIR} and ${BINDIR} which are expanded where the outputs are used.

###### Macro [PYTHON2\_MODULE][]() <a name="macro_PYTHON2_MODULE"></a>
Use in PY\_ANY\_MODULE to set it up for Python 2.x

###### Macro [PYTHON3\_ADDINCL][]() <a name="macro_PYTHON3_ADDINCL"></a>
This macro adds include path for Python headers (Python 3.x variant)
This should be used in 2 cases only:
- In PYMODULE since it compiles into .so and uses external Python runtime
- In system Python libraries themselves since peerdir there may create a loop

Never use this macro in PY3\_PROGRAM and PY3\_LIBRARY and PY23\_LIBRARY: they have everything by default
In all other cases use USE\_PYTHON3() macro instead

Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs

###### Macro PYTHON2\_MODULE() <a name="macro_PYTHON3_MODULE"></a>
Use in PY\_ANY\_MODULE to set it up for Python 3.x

###### Macro [PYTHON\_ADDINCL][]() <a name="macro_PYTHON_ADDINCL"></a>
This macro adds include path for Python headers (Python 2.x variant)
This should be used in 2 cases only:
- In PYMODULE since it compiles into .so and uses external Python runtime
- In system Python libraries themselves since proper PEERDIR there may create a loop

Never use this macro in PY\_PROGRAM, PY\_LIBRARY and PY23\_LIBRARY: they have everything by default
In all other cases use USE\_PYTHON macro instead

Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs

###### Macro [PYTHON\_PATH][](Path) <a name="macro_PYTHON_PATH"></a>
Not documented yet.

###### Macro [PY\_CODENAV][](Program) _# deprecated_ <a name="macro_PY_CODENAV"></a>
Collect codenavigation information from Program (specified as Program's module directory) and write it as Program.xref file
Most probably you'll never need this. If you think you need, please contact devtools@ for assistance.

###### Macro PY\_DOCTEST(Packages...) <a name="macro_PY_DOCTESTS"></a>
Add to the test doctests for specified Python packages
The packages should be part of a test (listed as sources of the test or its PEERDIRs).

###### Macro [PY\_MAIN][](package.module[:func]) <a name="macro_PY_MAIN"></a>
Specifies the module or function from which to start executing a python program

Documentation: https://wiki.yandex-team.ru/arcadia/python/pysrcs/#modulipyprogrampy3programimakrospymain

###### Macro [PY\_NAMESPACE][](prefix) <a name="macro_PY_NAMESPACE"></a>
Sets default Python namespace for all python sources in the module
Especially suitable in PROTO\_LIBRARY where Python sources are generated
and there is no PY\_SRCS to place NAMESPACE parameter

###### Macro [PY\_PROTOS\_FOR][](path/to/module)  _#builtin deprecated_ <a name="macro_PY_PROTOS_FOR"></a>
Deprecated.

Use PROTO\_LIBRARY() in order to have .proto compiled into Python.
Generates pb2.py files out of .proto files and saves those into PACKAGE

###### Macro [PY\_REGISTER][]([package.]module\_name[=module\_name]) <a name="macro_PY_REGISTER"></a>
Python knows about which built-ins can be imported, due to their registration in the Assembly or at the start of the interpreter.
All modules from the sources listed in PY\_SRCS() are registered automatically.
To register the modules from the sources in the SRCS(), you need to use PY\_REGISTER().

PY\_REGISTER(module\_name) initializes module globally via call to initmodule\_name()
PY\_REGISTER(package.module\_name) initializes module in the specified package using package-specific initialization using init7package11module\_name
PY\_REGISTER(package.module\_name=module\_name) redeclares global initialization of a module to use
package-specific initialization within package CFLAGS(-Dinitmodule\_name=init7package11module\_name)

PY\_REGISTER honors Python2 and Python3 differences and adjusts itself to Python version of a current module
Documentation: https://wiki.yandex-team.ru/arcadia/python/pysrcs/#makrospyregister

###### Macro [PY\_SRCS][]() <a name="macro_PY_SRCS"></a>
Not documented yet.

###### Macro [RECURSE][]([LOCAL] dirs...)  _# builtin_ <a name="macro_RECURSE"></a>
Add directories to the build
All projects must be reachable from the root chain RECURSE() for monorepo continuous integration functionality

###### Macro [RECURSE\_FOR\_TESTS][]([LOCAL] dirs...)  _# builtin_ <a name="macro_RECURSE_FOR_TESTS"></a>
Add directories to the build if tests are demanded

###### Macro [RECURSE\_ROOT\_RELATIVE][](dirlist)  _# builtin_ <a name="macro_RECURSE_ROOT_RELATIVE"></a>
In comparison with RECURSE(), in dirlist there must be a directory relative to the root (${ARCADIA\_ROOT})

###### Macro [REGISTER\_YQL\_PYTHON\_UDF][] <a name="macro_REGISTER_YQL_PYTHON_UDF"></a>
Not documented yet.

###### Macro [REQUIREMENTS][]([cpu:<count>] [disk\_usage:<size>] [ram:<size>] [ram\_disk:<size>] [container:<id>] [network:<restricted|full>] [dns:dns64]) <a name="macro_REQUIREMENTS"></a>
Allows you to specify the requirements of the test.

Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Macro [RESOURCE][](file name file name...) <a name="macro_RESOURCE"></a>
Add data (resources, random files) to the program)

This is a simpler but less flexible option than ARCHIVE(), because in the case of ARCHIVE(), you have to use the data explicitly,
and in the case of RESOURCE(), the data falls through SRCS(GLOBAL).
Therefore, there is no static data library from RESOURCE(), they are added only at the program linking stage.

@example: https://wiki.yandex-team.ru/yatool/howtowriteyamakefiles/#a2ispolzujjtekomanduresource
@example:
    LIBRARY()

    OWNER(user1)

    RESOURCE(
    path/to/file1 /key/in/program/1
    path/to/file2 /key2
    )

    END()

###### Macro [RESOURCE\_FILES][] <a name="macro_RESOURCE_FILES"></a>
RESOURCE\_FILES([PREFIX {prefix}] {path}) expands into
RESOURCE({path} resfs/file/{prefix}{path}
    - resfs/src/resfs/file/{prefix}{path}={rootrel\_arc\_src(path)}
)

resfs/src/{key} stores a source root (or build root) relative path of the
source of the value of the {key} resource.

resfs/file/{key} stores any value whose source was a file on a filesystem.
resfs/src/resfs/file/{key} must store its path.

This form is for use from other plugins:
RESOURCE\_FILES([DEST {dest}] {path}) expands into RESOURCE({path} resfs/file/{dest})

###### Macro [RUN][] <a name="macro_RUN"></a>
Not documented yet.

###### Macro [RUN\_ANTLR][] <a name="macro_RUN_ANTLR"></a>
Not documented yet.

###### Macro [RUN\_ANTLR4][] <a name="macro_RUN_ANTLR4"></a>
Not documented yet.

###### Macro [RUN\_ANTLR4\_CPP][] <a name="macro_RUN_ANTLR4_CPP"></a>
Not documented yet.

###### Macro [RUN\_JAVA][] <a name="macro_RUN_JAVA"></a>
Not documented yet.

###### Macro [RUN\_JAVA\_PROGRAM][] <a name="macro_RUN_JAVA_PROGRAM"></a>
Custom code generation
@link: https://wiki.yandex-team.ru/yatool/java/#kodogeneracijarunjavaprogram

###### Macro [RUN\_PROGRAM][](tool\_path args... [CWD dir] [ENV key=value...] [TOOL tools...] [IN inputs...] [OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output] [OUTPUT\_INCLUDES output\_includes...]) <a name="macro_RUN_PROGRAM"></a>
Run a program from arcadia.
These macros are similar: RUN\_PROGRAM, LUA, PYTHON, BUILTIN\_PYTHON.

Parameters:
- tool\_path - Path to the directory of the tool.
- args... - Program arguments. Relative paths listed in TOOL, IN, OUT, STDOUT become absolute.
- CWD dir - Absolute path of the working directory.
- ENV key=value... - Environment variables.
- TOOL tools... - Auxiliary tool directories.
- IN inputs... - Input files.
- OUT[\_NOAUTO] outputs... - Output files. NOAUTO outputs are not automatically added to the build process.
- STDOUT[\_NOAUTO] output - Redirect the standard output to the output file.
- OUTPUT\_INCLUDES output\_includes... - Includes of the output files that are needed to build them.

For absolute paths use ${ARCADIA\_ROOT} and ${ARCADIA\_BUILD\_ROOT}, or
${CURDIR} and ${BINDIR} which are expanded where the outputs are used.

###### Macro [RUN\_PYTHON][](Args...) <a name="macro_RUN_PYTHON"></a>
Version of RUN() macro to invoke Python scripts
@see: [RUN()](#macro\_RUN)

###### Macro [SET][](varname value)  _#builtin_ <a name="macro_SET"></a>
Sets varname to value

###### Macro [SETUP\_EXECTEST][] <a name="macro_SETUP_EXECTEST"></a>
Not documented yet.

###### Macro [SETUP\_PYTEST\_BIN][] <a name="macro_SETUP_PYTEST_BIN"></a>
Not documented yet.

###### Macro [SETUP\_RUN\_PYTHON][] <a name="macro_SETUP_RUN_PYTHON"></a>
Not documented yet.

###### Macro [SET\_APPEND][](varname appendvalue)  _#builtin_ <a name="macro_SET_APPEND"></a>
Appends appendvalue to varname's value using space as a separator

###### Macro [SET\_APPEND\_WITH\_GLOBAL][](varname appendvalue)  _#builtin_ <a name="macro_SET_APPEND_WITH_GLOBAL"></a>
Appends appendvalue to varname's value using space as a separator.
New value is propagated to dependants

###### Macro [SIZE][](SMALL/MEDIUM/LARGE) <a name="macro_SIZE"></a>
Marks a test of the specified size.

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [SKIP\_TEST][](Reason...) <a name="macro_SKIP_TEST"></a>
Not documented yet.

###### Macro [SOURCE\_GROUP][](...)  _#bultin_ <a name="macro_SOURCE_GROUP"></a>
Ignored

###### Macro [SPLIT\_CODEGEN][](tool prefix opts... [OUT\_NUM num] [OUTPUT\_INCLUDES output\_includes...]) <a name="macro_SPLIT_CODEGEN"></a>
Generator of a certain number of parts of the .cpp file + one header .h file from .in

Supports keywords:
1. OUT\_NUM <the number of generated Prefix.N.cpp default 25 (N varies from 0 to 24)>
2. OUTPUT\_INCLUDES <path to files that will be included in generalnyj of macro files>

###### Macro [SPLIT\_DWARF][]() <a name="macro_SPLIT_DWARF"></a>
Emit debug info for the PROGRAM as a separate file <module\_name>.debug
This may only be specified for programs. Use inside LIBRARY ALLOCATOR() END() leads to errors and stop the build.

###### Macro [SPLIT\_FACTOR][](x) <a name="macro_SPLIT_FACTOR"></a>
Sets the number of chunks for parallel run tests when using FORK\_TESTS, FORK\_SUBTESTS
As yourself use means the macro FORK\_TESTS.

Supports C++ ut and PyTest.

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [SRC][](File Flags...) <a name="macro_SRC"></a>
Compile single file with extra Flags.
Compilation is driven by the last extension of the File and Flags are specific to corresponding compilation command

###### Macro [SRCDIR][](dirlist)  _# builtin_ <a name="macro_SRCDIR"></a>
Add the specified directories to the list of those in which the source files will be searched
Available only for arcadia/contrib

###### Macro [SRCS][](<[GLOBAL] File> ...) <a name="macro_SRCS"></a>
Source files of the project
Arcadia Paths from the root and is relative to the project's LIST are supported

GLOBAL marks next file as direct input link of the program/shared library project built into
The scope of the GLOBAL keyword - the following file (that is, in the case of SRCS(GLOBAL foo.the cpp bar.cpp) global will be only foo.cpp)

@example:
Consider the file to ya.make:
```
    LIBRARY(test\_global)
    SRCS(GLOBAL foo.cpp)
    END()
```
SRCS(GLOBAL foo.cpp) - use to PROGRAM and DLL that is implicitly\* peerdir Yat test\_global had clearly been linked with foo.cpp.o.
- we say that prog1 (or dll1) implicitly peerdirs libN, if there is a
relationship of the form: "prog1 (or dll1) peerdirs lib1, lib1 peerdirs lib2,
..., libN-1 peerdirs libN"

###### Macro [SRC\_CPP\_AVX][](File Flags...) <a name="macro_SRC_CPP_AVX"></a>
Compile single .cpp-file with AVX and extra Flags

###### Macro [SRC\_CPP\_AVX2][](File Flags...) <a name="macro_SRC_CPP_AVX2"></a>
Compile single .cpp-file with AVX2 and extra Flags

###### Macro [SRC\_CPP\_SSE2][](File Flags...) <a name="macro_SRC_CPP_SSE2"></a>
Compile single .cpp-file with SSE2 and extra Flags

###### Macro [SRC\_CPP\_SSE3][](File Flags...) <a name="macro_SRC_CPP_SSE3"></a>
Compile single .cpp-file with SSE3 and extra Flags

###### Macro [SRC\_CPP\_SSE4][](File Flags...) <a name="macro_SRC_CPP_SSE4"></a>
Compile single .cpp-file with SSE4 (SSE4.1, SSE4.2 and POPCNT) and extra Flags

###### Macro [SRC\_CPP\_SSE41][](File Flags...) <a name="macro_SRC_CPP_SSE41"></a>
Compile single .cpp-file with SSE4.1 and extra Flags

###### Macro [SRC\_CPP\_SSSE3][](File Flags...) <a name="macro_SRC_CPP_SSSE3"></a>
Compile single .cpp-file with SSSE3 and extra Flags

###### Macro [SRC\_C\_AVX][](File Flags...) <a name="macro_SRC_C_AVX"></a>
Compile single .c-file with AVX and extra Flags

###### Macro [SRC\_C\_AVX2][](File Flags...) <a name="macro_SRC_C_AVX2"></a>
Compile single .c-file with AVX2 and extra Flags

###### Macro [SRC\_C\_SSE2][](File Flags...) <a name="macro_SRC_C_SSE2"></a>
Compile single .c-file with SSE2 and extra Flags

###### Macro [SRC\_C\_SSE3][](File Flags...) <a name="macro_SRC_C_SSE3"></a>
Compile single .c-file with SSE3 and extra Flags

###### Macro [SRC\_C\_SSE4][](File Flags...) <a name="macro_SRC_C_SSE4"></a>
Compile single .c-file with SSE4 (SSE4.1, SSE4.2 and POPCNT) and extra Flags

###### Macro [SRC\_C\_SSE41][](File Flags...) <a name="macro_SRC_C_SSE41"></a>
Compile single .c-file with SSE4.1 and extra Flags

###### Macro [SRC\_C\_SSSE3][](File Flags...) <a name="macro_SRC_C_SSSE3"></a>
Compile single .c-file with SSSE3 and extra Flags

###### Macro [STRIP][]() <a name="macro_STRIP"></a>
Strip debug info from a PROGRAM, DLL or TEST.
This macro doesn't work in LIBRARY's, UNION's and PACKAGE's

###### Macro [STRUCT\_CODEGEN][](Prefix) <a name="macro_STRUCT_CODEGEN"></a>
A special case BASE\_CODEGEN, in which the kernel/struct\_codegen/codegen\_tool tool is used

###### Macro [SYMLINK][](from to) <a name="macro_SYMLINK"></a>
Add symlink

###### Macro [SYSTEM\_PROPERTIES][]([<Key Value>...] [<File Path>...]) <a name="macro_SYSTEM_PROPERTIES"></a>
List of Key,Value pairs that will be available to test via System.getProperty.
FILE means that parst should be read from file specifies as Path
Documentation: https://wiki.yandex-team.ru/yatool/test/

###### Macro [TAG][] ([tag...]) <a name="macro_TAG"></a>
Each test can have one or more tags used to filter tests list for running.
There are also special tags affecting test behaviour, for example ya:external, sb:ssd.

Documentation: https://wiki.yandex-team.ru/yatool/test/#obshhieponjatija

###### Macro [TASKLET][]() <a name="macro_TASKLET"></a>
Not documented yet.

###### Macro [TASKLET\_REG][](Name, Lang, Impl, Includes...) <a name="macro_TASKLET_REG"></a>
Not documented yet.

###### Macro [TEST\_CWD][](path) <a name="macro_TEST_CWD"></a>
Defines cwd test. Used in conjunction with DATA()

Is only used inside of the TEST modules

###### Macro [TEST\_DATA][] <a name="macro_TEST_DATA"></a>
Not documented yet.

###### Macro [TEST\_SRCS][](Tests...) <a name="macro_TEST_SRCS"></a>
Not documented yet.

###### Macro [TGZ\_FOR][](path/to/archive)  _#builtin_ <a name="macro_TGZ_FOR"></a>
Not documented yet.

###### Macro [TIMEOUT][](TIMEOUT) <a name="macro_TIMEOUT"></a>
Sets a timeout on test execution

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [TOUCH][](Outputs...) <a name="macro_TOUCH"></a>
Not documented yet.

###### Macro [UBERJAR][]() <a name="macro_UBERJAR"></a>
UBERJAR is a single all-in-one jar-archive that includes all its Java dependencies (achievable PERDIR).
It also supports shading classes inside the archive by moving them to a different package (similar to the maven-shade-plugin).
Use UBERJAR inside JAVA\_PROGRAM module.

You can use the following macros to configure the archive:
1. UBERJAR\_HIDING\_PREFIX prefix for classes to shade (classes remain in their packages by default)
2. UBERJAR\_HIDE\_EXCLUDE\_PATTERN exculude classes matching this patterns from shading (if enabled).
3. UBERJAR\_PATH\_EXCLUDE\_PREFIX the prefix for classes that should not get into the jar archive (all classes are placed into the archive by default)
Documentation: https://wiki.yandex-team.ru/yatool/java/
@see: [JAVA\_PROGRAM](#module\_JAVA\_PROGRAM), [UBERJAR\_HIDING\_PREFIX](#macro\_UBERJAR\_HIDING\_PREFIX), [UBERJAR\_HIDE\_EXCLUDE\_PATTERN](#macro\_UBERJAR\_HIDE\_EXCLUDE\_PATTERN), [UBERJAR\_PATH\_EXCLUDE\_PREFIX](#macro\_UBERJAR\_PATH\_EXCLUDE\_PREFIX)

###### Macro [UBERJAR\_HIDE\_EXCLUDE\_PATTERN][](Args...) <a name="macro_UBERJAR_HIDE_EXCLUDE_PATTERN"></a>
Exculude classes matching this patterns from shading (if enabled)
Pattern may contain '\*' and '\*\*' globs
Shading is enabled for UBERJAR program using UBERJAR\_HIDING\_PREFIX macro. If this macro is not specified all classes are shaded.
@see: [UBERJAR](#macro\_UBERJAR), [UBERJAR\_HIDING\_PREFIX](#macro\_UBERJAR\_HIDING\_PREFIX)

###### Macro [UBERJAR\_HIDING\_PREFIX][](Arg) <a name="macro_UBERJAR_HIDING_PREFIX"></a>
Set prefix for classes to shade. All classes in UBERJAR will be moved into package prefixed with Arg
Classes remain in their packages by default
@see: [UBERJAR](#macro\_UBERJAR)

###### Macro [UBERJAR\_PATH\_EXCLUDE\_PREFIX][](Args...) <a name="macro_UBERJAR_PATH_EXCLUDE_PREFIX"></a>
Exculude classes matching this patterns from UBERJAR.
By default all dependencies of UBERJAR program will lend in a .jar archive.
@see: [UBERJAR](#macro\_UBERJAR)

###### Macro [USE\_ERROR\_PRONE][]() <a name="macro_USE_ERROR_PRONE"></a>
Use errorprone instead of javac for .java compilation

###### Macro [USE\_LINKER][]() <a name="macro_USE_LINKER"></a>
Not documented yet.

###### Macro [USE\_LINKER\_BFD][]() <a name="macro_USE_LINKER_BFD"></a>
Not documented yet.

###### Macro [USE\_LINKER\_GOLD][]() <a name="macro_USE_LINKER_GOLD"></a>
Not documented yet.

###### Macro [USE\_LINKER\_LLD][]() <a name="macro_USE_LINKER_LLD"></a>
Not documented yet.

###### Macro [USE\_PERL\_LIB][]() <a name="macro_USE_PERL_LIB"></a>
Add dependency on Perl to your LIBRARY

###### Macro [USE\_PYTHON][]() <a name="macro_USE_PYTHON"></a>
This adds Python 2.x runtime library to your LIBRARY and makes it Python2-compatible
If you'd like to use #include <Python.h> with Python2 specify USE\_PYTHON or better make it PY\_LIBRARY
If you'd like to use #include <Python.h> with Python3 specify USE\_PYTHON3 or better make it PY3\_LIBRARY
If you'd like to use #include <Python.h> with both Python2 and Python3 convert your LIBRARY to PY23\_LIBRARY
@see: [PY\_LIBRARY](#module\_PY\_LIBRARY), [PY3\_LIBRARY](#module\_PY3\_LIBRARY), [PY23\_LIBRARY](#multimodule\_PY23\_LIBRARY)

###### Macro [USE\_PYTHON3][]() <a name="macro_USE_PYTHON3"></a>
This adds Python3 library to your LIBRARY and makes it Python3-compatible
If you'd like to use #include <Python.h> with Python3 specify USE\_PYTHON3 or better make it PY3\_LIBRARY
If you'd like to use #include <Python.h> with Python2 specify USE\_PYTHON or better make it PY\_LIBRARY
If you'd like to use #include <Python.h> with both Python2 and Python3 convert your LIBRARY to PY23\_LIBRARY
@see: [PY\_LIBRARY](#module\_PY\_LIBRARY), [PY3\_LIBRARY](#module\_PY3\_LIBRARY), [PY23\_LIBRARY](#multimodule\_PY23\_LIBRARY)

###### Macro [USE\_RECIPE][](path [arg1 arg2...]) <a name="macro_USE_RECIPE"></a>
Provides prepared environment via recipe for test
Documentation: https://wiki.yandex-team.ru/yatool/test/recipes

###### Macro [USE\_YQL][]() <a name="macro_USE_YQL"></a>
Add required PEERDIRs for YQL module.
Can be used for in-place module definition in user LIBRARY.

###### Macro [VERSION][](Args...) <a name="macro_VERSION"></a>
Specify version of a module. Currently unused by build system, only informative.

###### Macro [WERROR][]() <a name="macro_WERROR"></a>
Consider warnings as errors in the current module.
In the bright future will be removed, since WERROR is the default.
Priorities: NO\_COMPILER\_WARNINGS > NO\_WERROR > WERROR\_MODE > WERROR

###### Macro [WITH\_JDK][]() <a name="macro_WITH_JDK"></a>
Not documented yet.

###### Macro [XS\_PROTO][](InputProto Dir Outputs...) _# deprecated_ <a name="macro_XS_PROTO"></a>
Generate Perl code from protobuf.
In order to use this macro one should predict all outputs protoc will emit from input\_proto file and enlist those as outputs

###### Macro [YABS\_GENERATE\_CONF][] <a name="macro_YABS_GENERATE_CONF"></a>
Not documented yet.

###### Macro [YABS\_GENERATE\_PHANTOM\_CONF\_PATCH][] <a name="macro_YABS_GENERATE_PHANTOM_CONF_PATCH"></a>
Not documented yet.

###### Macro [YABS\_GENERATE\_PHANTOM\_CONF\_TEST\_CHECK][] <a name="macro_YABS_GENERATE_PHANTOM_CONF_TEST_CHECK"></a>
Not documented yet.

###### Macro [YABS\_SERVER\_BUILD\_YSON\_INDEX][] <a name="macro_YABS_SERVER_BUILD_YSON_INDEX"></a>
Not documented yet.

###### Macro [YABS\_SERVER\_PREPARE\_QXL\_FROM\_SANDBOX][] <a name="macro_YABS_SERVER_PREPARE_QXL_FROM_SANDBOX"></a>
Not documented yet.

###### Macro [YDB\_GRPC][]() _# reserved_ <a name="macro_YDB_GRPC"></a>
Reserved for YDB specific GRPC plugin

###### Macro [YMAPS\_SPROTO][](ProtoFiles...) _# maps-specific_ <a name="macro_YMAPS_SPROTO"></a>
Maps-specific .proto handling: generate .sproto.h files using maps/libs/sproto/sprotoc

###### Macro [YQL\_ABI\_VERSION][](major minor release)) <a name="macro_YQL_ABI_VERSION"></a>
Specifying the supported ABI for YQL\_UDF.
@see: [YQL\_UDF()](#multimodule\_YQL\_UDF)

###### Macro [\_ADD\_PY\_PROTO\_OUT][](Suf) _#internal_ <a name="macro__ADD_PY_PROTO_OUT"></a>
Not documented yet.

###### Macro [\_ARCADIA\_PYTHON3\_ADDINCL][]()  _# internal_ <a name="macro__ARCADIA_PYTHON3_ADDINCL"></a>
This macro sets up Python3 headers for modules with Arcadia python (e.g. PY3\_LIBRARY) and configures module as Python 3.x

###### Macro [\_ARCADIA\_PYTHON\_ADDINCL][]()  _# internal_ <a name="macro__ARCADIA_PYTHON_ADDINCL"></a>
This macro sets up Python headers for modules with Arcadia python (e.g. PY\_LIBRARY) and configures module as Python 2.x

###### Macro [\_BUILDWITH\_CYTHON\_CPP\_DEP][](Src Dep Options...) _# internal_ <a name="macro__BUILDWITH_CYTHON_CPP_DEP"></a>
Generates .cpp file from .pyx and attach extra input Dep
If Dep changes the .cpp file will be re-generated

###### Macro [\_BUILDWITH\_CYTHON\_C\_API\_H][](Src Dep Options...) _# internal_ <a name="macro__BUILDWITH_CYTHON_C_API_H"></a>
BUILDWITH\_CYTHON\_C\_H with cdef api \_api.h file.

###### Macro [\_BUILDWITH\_CYTHON\_C\_DEP][](Src Dep Options...) _# internal_ <a name="macro__BUILDWITH_CYTHON_C_DEP"></a>
Generates .c file from .pyx and attach extra input Dep
If Dep changes the .c file will be re-generated

###### Macro [\_BUILDWITH\_CYTHON\_C\_H][](Src Dep Options...) _# internal_ <a name="macro__BUILDWITH_CYTHON_C_H"></a>
BUILDWITH\_CYTHON\_C without .pyx infix and with cdef public .h file.

###### Macro [\_BUILD\_MNS\_CPP][](NAME="", CHECK?, RANKING\_SUFFIX="", Files...) _#internal_ <a name="macro__BUILD_MNS_CPP"></a>
Not documented yet.

###### Macro [\_BUILD\_MNS\_FILE][](Input, Name, Output, Suffix, Check, Fml\_tool, AsmDataName) _#internal_ <a name="macro__BUILD_MNS_FILE"></a>
Not documented yet.

###### Macro [\_BUILD\_MNS\_FILES][] _#internal_ <a name="macro__BUILD_MNS_FILES"></a>
Not documented yet.

###### Macro [\_BUILD\_MNS\_HEADER][](NAME="", CHECK?, RANKING\_SUFFIX="", Files...) _#internal_ <a name="macro__BUILD_MNS_HEADER"></a>
Not documented yet.

###### Macro [\_BUNDLE\_TARGET][](Target, Destination) _#internal_ <a name="macro__BUNDLE_TARGET"></a>
Not documented yet.

###### Macro [\_CPP\_EVLOG\_CMD][](File) _#internal_ <a name="macro__CPP_EVLOG_CMD"></a>
Not documented yet.

###### Macro [\_CPP\_PROTO\_CMD][](File) _#internal_ <a name="macro__CPP_PROTO_CMD"></a>
Not documented yet.

###### Macro [\_CPP\_PROTO\_EVLOG\_CMD][](File) _#internal_ <a name="macro__CPP_PROTO_EVLOG_CMD"></a>
Not documented yet.

###### Macro [\_GENERATE\_PY\_EVS\_INTERNAL][](FILES...) _#internal_ <a name="macro__GENERATE_PY_EVS_INTERNAL"></a>
Not documented yet.

###### Macro [\_GENERATE\_PY\_PROTOS\_INTERNAL][](FILES...) _#internal_ <a name="macro__GENERATE_PY_PROTOS_INTERNAL"></a>
Not documented yet.

###### Macro [\_GO\_COMPILE\_CGO1][](NAME, FLAGS[], FILES...) _#internal_ <a name="macro__GO_COMPILE_CGO1"></a>
Not documented yet.

###### Macro [\_GO\_COMPILE\_CGO2][](NAME, C\_FILES[], S\_FILES[], FILES...) _#internal_ <a name="macro__GO_COMPILE_CGO2"></a>
Not documented yet.

###### Macro [\_GO\_COMPILE\_SYMABIS][](ASM\_FILES...) _#internal_ <a name="macro__GO_COMPILE_SYMABIS"></a>
Not documented yet.

###### Macro [\_GO\_FAKE\_OUTPUT][](FILES...) _#internal_ <a name="macro__GO_FAKE_OUTPUT"></a>
Not documented yet.

###### Macro [\_GO\_GRPC][]() _#internal_ <a name="macro__GO_GRPC"></a>
Not documented yet.

###### Macro [\_GO\_LINK\_EXE\_IMPL][](GO\_FILES...) _#internal_ <a name="macro__GO_LINK_EXE_IMPL"></a>
Not documented yet.

###### Macro [\_GO\_LINK\_LIB\_IMPL][](GO\_FILES...) _#internal_ <a name="macro__GO_LINK_LIB_IMPL"></a>
Not documented yet.

###### Macro [\_GO\_LINK\_TEST\_IMPL][](GO\_TEST\_FILES[], GO\_XTEST\_FILES[], GO\_FILES...) _#internal_ <a name="macro__GO_LINK_TEST_IMPL"></a>
Not documented yet.

###### Macro [\_GO\_PROCESS\_SRCS][] _#internal_ <a name="macro__GO_PROCESS_SRCS"></a>
\_GO\_PROCESS\_SRCS() macro processes only 'CGO' files. All remaining \*.go files
and other input files are currently processed by a link command of the
GO module (GO\_LIBRARY, GO\_PROGRAM)

###### Macro [\_GO\_PROTOC\_PLUGIN\_ARGS\_BASE][](Name, Tool, Plugins="") _#internal_ <a name="macro__GO_PROTOC_PLUGIN_ARGS_BASE"></a>
Not documented yet.

###### Macro [\_GO\_PROTO\_CMD][](File) _#internal_ <a name="macro__GO_PROTO_CMD"></a>
Not documented yet.

###### Macro [\_GO\_UNUSED\_SRCS][](FLAGS...) _#internal_ <a name="macro__GO_UNUSED_SRCS"></a>
Not documented yet.

###### Macro [\_GO\_YDB\_GRPC][](Plugins...) _#internal_ <a name="macro__GO_YDB_GRPC"></a>
Not documented yet.

###### Macro [\_IOS\_ASSETS][](AssetsDir, Content...) _#internal_ <a name="macro__IOS_ASSETS"></a>
Not documented yet.

###### Macro [\_JAVA\_EVLOG\_CMD][](File) _#internal_ <a name="macro__JAVA_EVLOG_CMD"></a>
Not documented yet.

###### Macro [\_JAVA\_PROTO\_CMD][](File) _#internal_ <a name="macro__JAVA_PROTO_CMD"></a>
Not documented yet.

###### Macro [\_LUAJIT\_21\_OBJDUMP][](Src, OUT="") _#internal_ <a name="macro__LUAJIT_21_OBJDUMP"></a>
Not documented yet.

###### Macro [\_LUAJIT\_OBJDUMP][](Src, OUT="") _#internal_ <a name="macro__LUAJIT_OBJDUMP"></a>
Not documented yet.

###### Macro [\_MAPKIT\_CPP\_PROTO\_CMD][](File) _#internal_ <a name="macro__MAPKIT_CPP_PROTO_CMD"></a>
Not documented yet.

###### Macro [\_MX\_BIN\_TO\_INFO][](Src) _#internal_ <a name="macro__MX_BIN_TO_INFO"></a>
Not documented yet.

###### Macro [\_MX\_GEN\_TABLE][](Srcs...) _#internal_ <a name="macro__MX_GEN_TABLE"></a>
Not documented yet.

###### Macro [\_PROTO\_PLUGIN\_ARGS\_BASE][](Name, Tool) _#internal_ <a name="macro__PROTO_PLUGIN_ARGS_BASE"></a>
Not documented yet.

###### Macro [\_PY3\_COMPILE\_BYTECODE][](SrcX Src) _# internal_ <a name="macro__PY3_COMPILE_BYTECODE"></a>
Compile Python 3.x .py source file into Arcadia binary form suitable for PY3\_PROGRAM
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/pysrcs/#makrospyregister

###### Macro [\_PY3\_REGISTER][]() _# internal_ <a name="macro__PY3_REGISTER"></a>
Register Python 3.x module in internal resource file system. Arcadia Python 3.x importer will be retrieve these on import directive
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/pysrcs/#makrospyregister

###### Macro \_PYTHON\_ADDINCL()  _# internal_ <a name="macro__PYTHON3_ADDINCL"></a>
This macro sets up Python 3.x headers for both Arcadia and non-Arcadia python

###### Macro [\_PYTHON\_ADDINCL][]()  _# internal_ <a name="macro__PYTHON_ADDINCL"></a>
This macro sets up Python 2.x headers for both Arcadia and non-Arcadia python

###### Macro [\_PY\_COMPILE\_BYTECODE][](SrcX Src) _# internal_ <a name="macro__PY_COMPILE_BYTECODE"></a>
Compile Python 2.x .py source file into Arcadia binary form suitable for PY\_PROGRAM
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/pysrcs/#makrospyregister

###### Macro [\_PY\_EVLOG\_CMD][](File) _#internal_ <a name="macro__PY_EVLOG_CMD"></a>
Not documented yet.

###### Macro [\_PY\_EVLOG\_CMD\_BASE][](File, Suf, Args...) _#internal_ <a name="macro__PY_EVLOG_CMD_BASE"></a>
Not documented yet.

###### Macro [\_PY\_EVLOG\_CMD\_INTERNAL][](File) _#internal_ <a name="macro__PY_EVLOG_CMD_INTERNAL"></a>
Not documented yet.

###### Macro [\_PY\_PROTO\_CMD][](File) _#internal_ <a name="macro__PY_PROTO_CMD"></a>
Not documented yet.

###### Macro [\_PY\_PROTO\_CMD\_BASE][](File, Suf, Args...) _#internal_ <a name="macro__PY_PROTO_CMD_BASE"></a>
Not documented yet.

###### Macro [\_PY\_PROTO\_CMD\_INTERNAL][](File) _#internal_ <a name="macro__PY_PROTO_CMD_INTERNAL"></a>
Not documented yet.

###### Macro [\_PY\_REGISTER][]() _# internal_ <a name="macro__PY_REGISTER"></a>
Register Python 2.x module in internal resource file system. Arcadia Python 2.x importer will be retrieve these on import directive
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/pysrcs/#makrospyregister

###### Macro [\_PY\_TEST][] _#internal_ <a name="macro__PY_TEST"></a>
Not documented yet.

###### Macro [\_RUN\_JAVA][](IN{input}[], OUT{output}[], OUT\_NOAUTO{output}[], OUTPUT\_INCLUDES[], TOOL[], STDOUT="", STDOUT\_NOAUTO="", CWD="", ENV[], HIDE\_OUTPUT?"stderr2stdout":"stdout2stderr", Args...) _#internal_ <a name="macro__RUN_JAVA"></a>
Not documented yet.

###### Macro [\_SPLIT\_CODEGEN\_BASE][](tool prefix OUTS[] OUTPUT\_INCLUDES[]) _# internal_ <a name="macro__SPLIT_CODEGEN_BASE"></a>
Generator of a certain number .the. cpp file + one header .h file from .in
This is the call of the generator. Python macro SPLIT\_CODEGEN() is defined in order to properly fill command outputs from OUT\_NUM argument

###### Macro [\_SRC][](Ext Src Flags) _# internal_ <a name="macro__SRC"></a>
Basic building block of extension-based command dispatching
To enable specific extension processing define \_SRC() macro with fixed first argument (Ext).
Internal logic will apply this macro to all files with this Ext listed in SRC/SRCS macros or outputs
of other commands (except ones marked as noauto)

###### Macro \_SRC("C", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____C"></a>
Not documented yet.

###### Macro \_SRC("S", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____S"></a>
Not documented yet.

###### Macro \_SRC("asm", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____asm"></a>
Not documented yet.

###### Macro \_SRC("asp", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____asp"></a>
Not documented yet.

###### Macro \_SRC("c", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____c"></a>
Not documented yet.

###### Macro \_SRC("cc", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____cc"></a>
Not documented yet.

###### Macro \_SRC("cpp", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____cpp"></a>
Not documented yet.

###### Macro \_SRC("cu", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____cu"></a>
Not documented yet.

###### Macro \_SRC("cxx", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____cxx"></a>
Not documented yet.

###### Macro \_SRC("ev", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____ev"></a>
Not documented yet.

###### Macro \_SRC("external", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____external"></a>
Not documented yet.

###### Macro \_SRC("f", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____f"></a>
Not documented yet.

###### Macro \_SRC("fml", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____fml"></a>
Not documented yet.

###### Macro \_SRC("fml2", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____fml2"></a>
Not documented yet.

###### Macro \_SRC("fml3", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____fml3"></a>
Not documented yet.

###### Macro \_SRC("gperf", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____gperf"></a>
Not documented yet.

###### Macro \_SRC("gztproto", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____gztproto"></a>
Not documented yet.

###### Macro \_SRC("in", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____in"></a>
Not documented yet.

###### Macro \_SRC("l", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____l"></a>
Not documented yet.

###### Macro \_SRC("lex", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____lex"></a>
Not documented yet.

###### Macro \_SRC("lpp", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____lpp"></a>
Not documented yet.

###### Macro \_SRC("m", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____m"></a>
Not documented yet.

###### Macro \_SRC("masm", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____masm"></a>
Not documented yet.

###### Macro \_SRC("mm", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____mm"></a>
Not documented yet.

###### Macro \_SRC("pln", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____pln"></a>
Not documented yet.

###### Macro \_SRC("po", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____po"></a>
Not documented yet.

###### Macro \_SRC("proto", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____proto"></a>
Not documented yet.

###### Macro \_SRC("pyx", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____pyx"></a>
Not documented yet.

###### Macro \_SRC("rl", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____rl"></a>
Not documented yet.

###### Macro \_SRC("rl5", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____rl5"></a>
Not documented yet.

###### Macro \_SRC("rl6", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____rl6"></a>
Not documented yet.

###### Macro \_SRC("s", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____s"></a>
Not documented yet.

###### Macro \_SRC("sc", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____sc"></a>
Not documented yet.

###### Macro \_SRC("sfdl", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____sfdl"></a>
Not documented yet.

###### Macro \_SRC("storyboard", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____storyboard"></a>
Not documented yet.

###### Macro \_SRC("supp", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____supp"></a>
Not documented yet.

###### Macro \_SRC("xib", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____xib"></a>
Not documented yet.

###### Macro \_SRC("xs", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____xs"></a>
Not documented yet.

###### Macro \_SRC("xsyn", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____xsyn"></a>
Not documented yet.

###### Macro \_SRC("y", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____y"></a>
Not documented yet.

###### Macro \_SRC("yasm", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____yasm"></a>
Not documented yet.

###### Macro \_SRC("ydl", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____ydl"></a>
Not documented yet.

###### Macro \_SRC("ypp", SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC____ypp"></a>
Not documented yet.

###### Macro [\_SRC\_c][](SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC_c"></a>
Not documented yet.

###### Macro [\_SRC\_cpp][](SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC_cpp"></a>
Not documented yet.

###### Macro [\_SRC\_m][](SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC_m"></a>
Not documented yet.

###### Macro [\_SRC\_masm][](SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC_masm"></a>
Not documented yet.

###### Macro [\_SRC\_yasm][](SRC, SRCFLAGS...) _#internal_ <a name="macro__SRC_yasm"></a>
Not documented yet.

###### Macro [\_SRC\_yasm\_impl][](SRC, PREINCLUDES[], SRCFLAGS...) _#internal_ <a name="macro__SRC_yasm_impl"></a>
Not documented yet.

###### Macro [\_TEST][] _#internal_ <a name="macro__TEST"></a>
Not documented yet.

###### Macro [\_UNITTEST][] _#internal_ <a name="macro__UNITTEST"></a>
Not documented yet.

###### Macro [\_YCR\_GENERATE\_CONFIGS][](Package, App) _#internal_ <a name="macro__YCR_GENERATE_CONFIGS"></a>
Not documented yet.

###### Macro [\_YCR\_GENERATE\_CONFIGS\_INTL][](Package, App, Configs...) _#internal_ <a name="macro__YCR_GENERATE_CONFIGS_INTL"></a>
Not documented yet.

###### Macro [\_YMAPS\_GENERATE\_SPROTO\_HEADER][](File) _#internal_ <a name="macro__YMAPS_GENERATE_SPROTO_HEADER"></a>
Not documented yet.

###### Macro [\_YTEST][] _#internal_ <a name="macro__YTEST"></a>
Not documented yet.

 [DOCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1892
 [PROTO\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4206
 [PY23\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4619
 [PY23\_NATIVE\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4640
 [SANDBOX\_TASK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2430
 [YQL\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1741
 [BENCHMARK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1291
 [BOOSTTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1029
 [DEV\_DLL\_PROXY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1595
 [DLL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1573
 [DLL\_JAVA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1620
 [DLL\_UNIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1427
 [EXECTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1279
 [FAT\_OBJECT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1411
 [FUZZ]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1004
 [GO\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4543
 [GO\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4557
 [GO\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4590
 [GTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1229
 [IOS\_APP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4695
 [JAVA\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2361
 [JAVA\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2355
 [JTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2372
 [JTEST\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2414
 [JUNIT5]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2401
 [LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1371
 [LINK\_UNIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L762
 [METAQUERY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2106
 [PACKAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1833
 [PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L907
 [PY3TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1217
 [PY3TEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1201
 [PY3\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2017
 [PY3\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L927
 [PYMODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1518
 [PYTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1195
 [PYTEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1181
 [PY\_ANY\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1478
 [PY\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1993
 [PY\_PACKAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1878
 [PY\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L916
 [RESOURCES\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1389
 [R\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1550
 [TESTNG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2388
 [TGZ\_PACKAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1851
 [UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1641
 [UDF\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1628
 [UDF\_LIB]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1649
 [UNION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1867
 [UNITTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L977
 [UNITTEST\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1313
 [UNITTEST\_WITH\_CUSTOM\_ENTRY\_POINT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L991
 [YCR\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L946
 [YQL\_PYTHON3\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1784
 [YQL\_PYTHON\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1763
 [YQL\_PYTHON\_UDF\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1805
 [YQL\_UDF\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1664
 [YQL\_UDF\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1242
 [YT\_UNITTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L984
 [\_BASE\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L866
 [\_BASE\_PY3\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2073
 [\_BASE\_PYTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1161
 [\_BASE\_PY\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2041
 [\_BASE\_UNIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L545
 [\_BASE\_UNITTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L962
 [\_DLL\_COMPATIBLE\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1735
 [\_GO\_BASE\_UNIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4483
 [\_JAVA\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1960
 [\_JAVA\_PLACEHOLDER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2334
 [\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1323
 [ACCELEO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/scarab_cant_clash.py#L4
 [ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [ADDINCLSELF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/addinclself.py#L2
 [ADD\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L411
 [ADD\_CHECK\_PY\_IMPORTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L468
 [ADD\_COMPILABLE\_TRANSLATE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2663
 [ADD\_COMPILABLE\_TRANSLIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2672
 [ADD\_DLLS\_TO\_JAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2289
 [ADD\_PERL\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2581
 [ADD\_PYTEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L530
 [ADD\_PYTEST\_SCRIPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L504
 [ADD\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L367
 [ADD\_WAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2208
 [ADD\_YTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L309
 [ALLOCATOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2606
 [ALL\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1725
 [ANNOTATION\_PROCESSOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2245
 [ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3296
 [ARCHIVE\_ASM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3277
 [ARCHIVE\_BY\_KEYS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3300
 [ASM\_PREINCLUDE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4114
 [BASE\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3373
 [BUILDWITH\_CYTHON\_C]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3200
 [BUILDWITH\_CYTHON\_CPP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3183
 [BUILDWITH\_RAGEL6]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3233
 [BUILD\_CATBOOST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3693
 [BUILD\_MN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3655
 [BUILD\_MNS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3679
 [BUILD\_ONLY\_IF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [BUILD\_PLNS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3701
 [BUILTIN\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3885
 [BUNDLE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/bundle.py#L4
 [CFG\_VARS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3359
 [CFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3430
 [CGO\_CFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4456
 [CGO\_LDFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4464
 [CGO\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4441
 [CHECK\_CONFIG\_H]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L300
 [CHECK\_DEPENDENT\_DIRS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [CHECK\_JAVA\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2135
 [CLANG\_EMIT\_AST\_CXX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4004
 [COMPILE\_C\_AS\_CXX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3921
 [COMPILE\_SWIFT\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4731
 [CONDITIONAL\_PEERDIR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/conditional_peerdir.py#L4
 [CONFIGURE\_FILE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3366
 [CONLYFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3443
 [COPY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/cp.py#L6
 [COPY\_FILE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2709
 [COPY\_FILE\_WITH\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2719
 [CREATE\_BUILDINFO\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3342
 [CREATE\_GO\_SVNVERSION\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3335
 [CREATE\_INIT\_PY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/create_init_py.py#L6
 [CREATE\_INIT\_PY\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/create_init_py.py#L40
 [CREATE\_INIT\_PY\_STRUCTURE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/create_init_py.py#L29
 [CREATE\_JAVA\_SVNVERSION\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3326
 [CREATE\_SVNVERSION\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3317
 [CTEMPLATE\_VARNAMES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3967
 [CUDA\_NVCC\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3455
 [CXXFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3449
 [DARWIN\_SIGNED\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4687
 [DARWIN\_STRINGS\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4683
 [DATA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1080
 [DEB\_VERSION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3645
 [DECIMAL\_MD5\_LOWER\_32\_BITS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3353
 [DECLARE\_EXTERNAL\_HOST\_RESOURCES\_BUNDLE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [DECLARE\_EXTERNAL\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [DEFAULT]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [DEPENDENCY\_MANAGEMENT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2325
 [DEPENDS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1111
 [DISABLE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [DLL\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [DOCS\_CONFIG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1931
 [DOCS\_DIR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1921
 [DOCS\_VARS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1941
 [DUMPERF\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3402
 [ELSE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [ELSEIF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [ENABLE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [END]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [ENDIF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [ENV]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1119
 [EXCLUDE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2253
 [EXCLUDE\_TAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [EXPORTS\_SCRIPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L955
 [EXPORT\_MAPKIT\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4150
 [EXPORT\_YMAPS\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3943
 [EXTERNAL\_JAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2298
 [EXTERNAL\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [EXTRADIR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [EXTRALIBS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [EXTRALIBS\_STATIC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2657
 [FAT\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/res.py#L31
 [FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/files.py#L1
 [FLAT\_JOIN\_SRCS\_GLOBAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2813
 [FORK\_SUBTESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2757
 [FORK\_TESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2745
 [FORK\_TEST\_FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2779
 [FROM\_MDS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3909
 [FROM\_SANDBOX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3899
 [FUZZ\_DICTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1044
 [FUZZ\_OPTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1061
 [GENERATED\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3995
 [GENERATE\_ENUM\_SERIALIZATION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3628
 [GENERATE\_ENUM\_SERIALIZATION\_WITH\_HEADER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3637
 [GENERATE\_PY\_PROTOS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2688
 [GENERATE\_SCRIPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/java.py#L50
 [GEN\_SCHEEME2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3791
 [GLOBAL\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1706
 [GO\_ASM\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4316
 [GO\_CGO1\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4323
 [GO\_CGO2\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4330
 [GO\_COMPILE\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4337
 [GO\_FAKE\_OUTPUT]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/plugins/plugin_go_fake_output_handler.cpp#L106
 [GO\_LDFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4449
 [GO\_LINK\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4344
 [GO\_PACKAGE\_NAME]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4405
 [GO\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4413
 [GO\_TEST\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [GO\_TEST\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4421
 [GO\_XTEST\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4429
 [GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L491
 [IDEA\_EXCLUDE\_DIRS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2215
 [IDEA\_RESOURCE\_DIRS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2222
 [IF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [IMPORT\_YMAPS\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3932
 [INCLUDE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [INCLUDE\_TAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [INDUCED\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [IOS\_APP\_ASSETS\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4679
 [IOS\_APP\_COMMON\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4674
 [IOS\_APP\_SETTINGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ios_app_settings.py#L5
 [IOS\_ASSETS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ios_assets.py#L6
 [JAVAC\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2229
 [JAVA\_IGNORE\_CLASSPATH\_CLASH\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4605
 [JAVA\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/java.py#L67
 [JAVA\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2284
 [JAVA\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L595
 [JAVA\_TEST\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L655
 [JOINSRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3569
 [JOIN\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2796
 [JOIN\_SRCS\_GLOBAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2804
 [JVM\_ARGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2127
 [LDFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3411
 [LDFLAGS\_FIXED]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3421
 [LICENSE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3738
 [LINT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1154
 [LJ\_21\_ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/lj_archive.py#L23
 [LJ\_ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/lj_archive.py#L1
 [LLVM\_BC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/llvm_bc.py#L6
 [LLVM\_COMPILE\_C]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4022
 [LLVM\_COMPILE\_CXX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4013
 [LLVM\_COMPILE\_LL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4031
 [LLVM\_LINK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4040
 [LLVM\_OPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4049
 [LSAN\_SUPPRESSIONS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/lsan_suppressions.py#L1
 [LUA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3841
 [MACROS\_WITH\_ERROR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/macros_with_error.py#L4
 [MAPKITIDL]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/plugins/plugin_mapkitidl_handler.cpp#L396
 [MAPKIT\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4128
 [MAPKIT\_ENABLE\_WHOLE\_ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4165
 [MAPSMOBI\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4143
 [MASMFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3436
 [MAVEN\_GROUP\_ID]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2236
 [MESSAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [METAQUERYFILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/mq.py#L12
 [MSVC\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4853
 [MX\_FORMULAS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/mx_archive.py#L1
 [NEED\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3711
 [NEED\_REVIEW]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3728
 [NO\_BUILD\_IF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [NO\_CHECK\_IMPORTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4082
 [NO\_CLANG\_COVERAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3590
 [NO\_CODENAVIGATION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3513
 [NO\_COMPILER\_WARNINGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3475
 [NO\_DEBUG\_INFO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3925
 [NO\_GPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3747
 [NO\_JOIN\_SRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3563
 [NO\_LIBC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3543
 [NO\_LINT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1147
 [NO\_NEED\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3719
 [NO\_OPTIMIZE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3468
 [NO\_OPTIMIZE\_PY\_PROTOS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L343
 [NO\_PLATFORM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3555
 [NO\_PLATFORM\_RESOURCES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3506
 [NO\_PYTHON\_INCLUDES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2450
 [NO\_RUNTIME]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3532
 [NO\_SANITIZE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3576
 [NO\_SANITIZE\_COVERAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3583
 [NO\_SSE4]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2878
 [NO\_UTIL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3521
 [NO\_WERROR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3493
 [NO\_WSHADOW]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3499
 [ONLY\_TAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [OPTIMIZE\_PY\_PROTOS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L334
 [OWNER]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [PACK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1817
 [PACKAGE\_STRICT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1822
 [PARTITIONED\_RECURSE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [PARTITIONED\_RECURSE\_FOR\_TESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [PARTITIONED\_RECURSE\_ROOT\_RELATIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [PEERDIR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [PIRE\_INLINE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3286
 [PIRE\_INLINE\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3282
 [PRINT\_MODULE\_TYPE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/print_module_type.py#L1
 [PROCESS\_DOCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/docs.py#L30
 [PROCESS\_DOCSLIB]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/docs.py#L26
 [PROTO2FBS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L435
 [PROVIDES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2821
 [PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3863
 [PYTHON2\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1526
 [PYTHON3\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2507
 [PYTHON3\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1537
 [PYTHON\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2465
 [PYTHON\_PATH]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1134
 [PY\_CODENAV]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4091
 [PY\_DOCTESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/pybuild.py#L435
 [PY\_MAIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/pybuild.py#L491
 [PY\_NAMESPACE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1951
 [PY\_PROTOS\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [PY\_REGISTER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/pybuild.py#L453
 [PY\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4609
 [RECURSE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [RECURSE\_FOR\_TESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [RECURSE\_ROOT\_RELATIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [REGISTER\_YQL\_PYTHON\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yql_python_udf.py#L10
 [REQUIREMENTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1101
 [RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/plugins/plugin_resource_handler.cpp#L42
 [RESOURCE\_FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/res.py#L47
 [RUN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L775
 [RUN\_ANTLR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/antlr.py#L4
 [RUN\_ANTLR4]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/antlr4.py#L6
 [RUN\_ANTLR4\_CPP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/antlr4.py#L18
 [RUN\_JAVA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/run_java.py#L5
 [RUN\_JAVA\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/java.py#L33
 [RUN\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3819
 [RUN\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4122
 [SET]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [SETUP\_EXECTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L781
 [SETUP\_PYTEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L765
 [SETUP\_RUN\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L789
 [SET\_APPEND]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [SET\_APPEND\_WITH\_GLOBAL]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [SIZE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2788
 [SKIP\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1139
 [SOURCE\_GROUP]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [SPLIT\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/split_codegen.py#L9
 [SPLIT\_DWARF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2642
 [SPLIT\_FACTOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2768
 [SRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3066
 [SRCDIR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3089
 [SRC\_CPP\_AVX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3169
 [SRC\_CPP\_AVX2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3175
 [SRC\_CPP\_SSE2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3139
 [SRC\_CPP\_SSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3145
 [SRC\_CPP\_SSE4]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3163
 [SRC\_CPP\_SSE41]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3157
 [SRC\_CPP\_SSSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3151
 [SRC\_C\_AVX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3127
 [SRC\_C\_AVX2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3133
 [SRC\_C\_SSE2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3097
 [SRC\_C\_SSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3103
 [SRC\_C\_SSE4]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3121
 [SRC\_C\_SSE41]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3115
 [SRC\_C\_SSSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3109
 [STRIP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3462
 [STRUCT\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3395
 [SYMLINK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3797
 [SYSTEM\_PROPERTIES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2119
 [TAG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1091
 [TASKLET]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4171
 [TASKLET\_REG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4190
 [TEST\_CWD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2152
 [TEST\_DATA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py#L34
 [TEST\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1066
 [TGZ\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp#L18
 [TIMEOUT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2731
 [TOUCH]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4065
 [UBERJAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2169
 [UBERJAR\_HIDE\_EXCLUDE\_PATTERN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2188
 [UBERJAR\_HIDING\_PREFIX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2178
 [UBERJAR\_PATH\_EXCLUDE\_PREFIX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2197
 [USE\_ERROR\_PRONE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2142
 [USE\_LINKER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4879
 [USE\_LINKER\_BFD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4895
 [USE\_LINKER\_GOLD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4898
 [USE\_LINKER\_LLD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4901
 [USE\_PERL\_LIB]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2575
 [USE\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2547
 [USE\_PYTHON3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2559
 [USE\_RECIPE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1128
 [USE\_YQL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1685
 [VERSION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3754
 [WERROR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3484
 [WITH\_JDK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2330
 [XS\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L426
 [YABS\_GENERATE\_CONF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_generate_conf.py#L10
 [YABS\_GENERATE\_PHANTOM\_CONF\_PATCH]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_generate_conf.py#L35
 [YABS\_GENERATE\_PHANTOM\_CONF\_TEST\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_generate_conf.py#L53
 [YABS\_SERVER\_BUILD\_YSON\_INDEX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_server_build_index.py#L121
 [YABS\_SERVER\_PREPARE\_QXL\_FROM\_SANDBOX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_server_build_index.py#L92
 [YDB\_GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L522
 [YMAPS\_SPROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3956
 [YQL\_ABI\_VERSION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L1754
 [\_ADD\_PY\_PROTO\_OUT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L351
 [\_ARCADIA\_PYTHON3\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2515
 [\_ARCADIA\_PYTHON\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2473
 [\_BUILDWITH\_CYTHON\_CPP\_DEP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3192
 [\_BUILDWITH\_CYTHON\_C\_API\_H]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3225
 [\_BUILDWITH\_CYTHON\_C\_DEP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3209
 [\_BUILDWITH\_CYTHON\_C\_H]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3217
 [\_BUILD\_MNS\_CPP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3665
 [\_BUILD\_MNS\_FILE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3659
 [\_BUILD\_MNS\_FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/build_mn_files.py#L4
 [\_BUILD\_MNS\_HEADER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3669
 [\_BUNDLE\_TARGET]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2723
 [\_CPP\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L388
 [\_CPP\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L383
 [\_CPP\_PROTO\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L393
 [\_GENERATE\_PY\_EVS\_INTERNAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2700
 [\_GENERATE\_PY\_PROTOS\_INTERNAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2694
 [\_GO\_COMPILE\_CGO1]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4366
 [\_GO\_COMPILE\_CGO2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4370
 [\_GO\_COMPILE\_SYMABIS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4362
 [\_GO\_FAKE\_OUTPUT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4392
 [\_GO\_GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4472
 [\_GO\_LINK\_EXE\_IMPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4378
 [\_GO\_LINK\_LIB\_IMPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4374
 [\_GO\_LINK\_TEST\_IMPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4382
 [\_GO\_PROCESS\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/gobuild.py#L28
 [\_GO\_PROTOC\_PLUGIN\_ARGS\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L374
 [\_GO\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L378
 [\_GO\_UNUSED\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4434
 [\_GO\_YDB\_GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4477
 [\_IOS\_ASSETS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4670
 [\_JAVA\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L456
 [\_JAVA\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L417
 [\_LUAJIT\_21\_OBJDUMP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3600
 [\_LUAJIT\_OBJDUMP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3595
 [\_MAPKIT\_CPP\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L399
 [\_MX\_BIN\_TO\_INFO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3604
 [\_MX\_GEN\_TABLE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3616
 [\_PROTO\_PLUGIN\_ARGS\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L347
 [\_PY3\_COMPILE\_BYTECODE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3268
 [\_PY3\_REGISTER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3251
 [\_PYTHON3\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2523
 [\_PYTHON\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2481
 [\_PY\_COMPILE\_BYTECODE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3260
 [\_PY\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L448
 [\_PY\_EVLOG\_CMD\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L443
 [\_PY\_EVLOG\_CMD\_INTERNAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L452
 [\_PY\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L409
 [\_PY\_PROTO\_CMD\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L405
 [\_PY\_PROTO\_CMD\_INTERNAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L413
 [\_PY\_REGISTER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3242
 [\_PY\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest2.py#L56
 [\_RUN\_JAVA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3889
 [\_SPLIT\_CODEGEN\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3381
 [\_SRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2888
 [\_SRC\_\_\_\_C]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3025
 [\_SRC\_\_\_\_S]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2892
 [\_SRC\_\_\_\_asm]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3045
 [\_SRC\_\_\_\_asp]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2945
 [\_SRC\_\_\_\_c]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3029
 [\_SRC\_\_\_\_cc]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3021
 [\_SRC\_\_\_\_cpp]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3013
 [\_SRC\_\_\_\_cu]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4963
 [\_SRC\_\_\_\_cxx]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3017
 [\_SRC\_\_\_\_ev]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2974
 [\_SRC\_\_\_\_external]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3003
 [\_SRC\_\_\_\_f]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3007
 [\_SRC\_\_\_\_fml]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2904
 [\_SRC\_\_\_\_fml2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2908
 [\_SRC\_\_\_\_fml3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2912
 [\_SRC\_\_\_\_gperf]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2933
 [\_SRC\_\_\_\_gztproto]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2982
 [\_SRC\_\_\_\_in]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2995
 [\_SRC\_\_\_\_l]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2953
 [\_SRC\_\_\_\_lex]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2958
 [\_SRC\_\_\_\_lpp]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2962
 [\_SRC\_\_\_\_m]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3033
 [\_SRC\_\_\_\_masm]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3037
 [\_SRC\_\_\_\_mm]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2900
 [\_SRC\_\_\_\_pln]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2966
 [\_SRC\_\_\_\_po]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3049
 [\_SRC\_\_\_\_proto]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2978
 [\_SRC\_\_\_\_pyx]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2989
 [\_SRC\_\_\_\_rl]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2937
 [\_SRC\_\_\_\_rl5]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2941
 [\_SRC\_\_\_\_rl6]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2949
 [\_SRC\_\_\_\_s]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2896
 [\_SRC\_\_\_\_sc]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2999
 [\_SRC\_\_\_\_sfdl]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2916
 [\_SRC\_\_\_\_storyboard]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4661
 [\_SRC\_\_\_\_supp]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3053
 [\_SRC\_\_\_\_xib]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4665
 [\_SRC\_\_\_\_xs]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2920
 [\_SRC\_\_\_\_xsyn]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2970
 [\_SRC\_\_\_\_y]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2924
 [\_SRC\_\_\_\_yasm]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3041
 [\_SRC\_\_\_\_ydl]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3057
 [\_SRC\_\_\_\_ypp]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L2929
 [\_SRC\_c]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4869
 [\_SRC\_cpp]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4866
 [\_SRC\_m]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4872
 [\_SRC\_masm]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4875
 [\_SRC\_yasm]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4107
 [\_SRC\_yasm\_impl]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L4943
 [\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest2.py#L60
 [\_UNITTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest2.py#L43
 [\_YCR\_GENERATE\_CONFIGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L937
 [\_YCR\_GENERATE\_CONFIGS\_INTL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L933
 [\_YMAPS\_GENERATE\_SPROTO\_HEADER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf#L3948
 [\_YTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest2.py#L52
