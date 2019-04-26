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
       - Module [EXECTEST](#module_EXECTEST)
       - Module [FAT_OBJECT](#module_FAT_OBJECT)
       - Module [FUZZ](#module_FUZZ)
       - Module [GO_BASE_UNIT](#module_GO_BASE_UNIT)
       - Module [GO_LIBRARY](#module_GO_LIBRARY)
       - Module [GO_PROGRAM](#module_GO_PROGRAM)
       - Module [GO_TEST](#module_GO_TEST)
       - Module [GTEST](#module_GTEST)
       - Module [IOS_APP](#module_IOS_APP)
       - Module [JAVA_LIBRARY](#module_JAVA_LIBRARY)
       - Module [JAVA_PLACEHOLDER](#module_JAVA_PLACEHOLDER)
       - Module [JAVA_PROGRAM](#module_JAVA_PROGRAM)
       - Module [JTEST](#module_JTEST)
       - Module [JTEST_FOR](#module_JTEST_FOR)
       - Module [JUNIT5](#module_JUNIT5)
       - Module [LIBRARY](#module_LIBRARY)
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
   * [Macros](#macros)
       - Macros [ACCELEO](#macro_ACCELEO) .. [ADD_PERL_MODULE](#macro_ADD_PERL_MODULE)
       - Macros [ADD_PYTEST_BIN](#macro_ADD_PYTEST_BIN) .. [ARCHIVE](#macro_ARCHIVE)
       - Macros [ARCHIVE_ASM](#macro_ARCHIVE_ASM) .. [BUILDWITH_CYTHON_C_H](#macro_BUILDWITH_CYTHON_C_H)
       - Macros [BUILDWITH_RAGEL6](#macro_BUILDWITH_RAGEL6) .. [BUILD_PLNS](#macro_BUILD_PLNS)
       - Macros [BUILTIN_PYTHON](#macro_BUILTIN_PYTHON) .. [CHECK_CONFIG_H](#macro_CHECK_CONFIG_H)
       - Macros [CHECK_DEPENDENT_DIRS](#macro_CHECK_DEPENDENT_DIRS) .. [COPY_FILE](#macro_COPY_FILE)
       - Macros [COPY_FILE_WITH_DEPS](#macro_COPY_FILE_WITH_DEPS) .. [CREATE_JAVA_SVNVERSION_FOR](#macro_CREATE_JAVA_SVNVERSION_FOR)
       - Macros [CREATE_SVNVERSION_FOR](#macro_CREATE_SVNVERSION_FOR) .. [DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE](#macro_DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE)
       - Macros [DECLARE_EXTERNAL_RESOURCE](#macro_DECLARE_EXTERNAL_RESOURCE) .. [DUMPERF_CODEGEN](#macro_DUMPERF_CODEGEN)
       - Macros [ELSE](#macro_ELSE) .. [EXPORT_MAPKIT_PROTO](#macro_EXPORT_MAPKIT_PROTO)
       - Macros [EXPORT_YMAPS_PROTO](#macro_EXPORT_YMAPS_PROTO) .. [FORK_SUBTESTS](#macro_FORK_SUBTESTS)
       - Macros [FORK_TESTS](#macro_FORK_TESTS) .. [GENERATE_PY_EVS](#macro_GENERATE_PY_EVS)
       - Macros [GENERATE_PY_PROTOS](#macro_GENERATE_PY_PROTOS) .. [GO_COMPILE_FLAGS](#macro_GO_COMPILE_FLAGS)
       - Macros [GO_COMPILE_SYMABIS](#macro_GO_COMPILE_SYMABIS) .. [GO_PROTOC_PLUGIN_ARGS_BASE](#macro_GO_PROTOC_PLUGIN_ARGS_BASE)
       - Macros [GO_PROTO_CMD](#macro_GO_PROTO_CMD) .. [IDEA_RESOURCE_DIRS](#macro_IDEA_RESOURCE_DIRS)
       - Macros [IF](#macro_IF) .. [JAVAC_FLAGS](#macro_JAVAC_FLAGS)
       - Macros [JAVA_EVLOG_CMD](#macro_JAVA_EVLOG_CMD) .. [JOIN_SRCS_GLOBAL](#macro_JOIN_SRCS_GLOBAL)
       - Macros [JVM_ARGS](#macro_JVM_ARGS) .. [LLVM_COMPILE_CXX](#macro_LLVM_COMPILE_CXX)
       - Macros [LLVM_COMPILE_LL](#macro_LLVM_COMPILE_LL) .. [MAPKIT_ADDINCL](#macro_MAPKIT_ADDINCL)
       - Macros [MAPKIT_CPP_PROTO_CMD](#macro_MAPKIT_CPP_PROTO_CMD) .. [MX_FORMULAS](#macro_MX_FORMULAS)
       - Macros [MX_GEN_TABLE](#macro_MX_GEN_TABLE) .. [NO_GPL](#macro_NO_GPL)
       - Macros [NO_JOIN_SRC](#macro_NO_JOIN_SRC) .. [NO_RUNTIME](#macro_NO_RUNTIME)
       - Macros [NO_SANITIZE](#macro_NO_SANITIZE) .. [PACK](#macro_PACK)
       - Macros [PACKAGE_STRICT](#macro_PACKAGE_STRICT) .. [PROCESS_DOCSLIB](#macro_PROCESS_DOCSLIB)
       - Macros [PROTO2FBS](#macro_PROTO2FBS) .. [PYTHON_PATH](#macro_PYTHON_PATH)
       - Macros [PY_CODENAV](#macro_PY_CODENAV) .. [PY_PROTO_CMD_BASE](#macro_PY_PROTO_CMD_BASE)
       - Macros [PY_REGISTER](#macro_PY_REGISTER) .. [RUN](#macro_RUN)
       - Macros [RUN_ANTLR](#macro_RUN_ANTLR) .. [SETUP_PYTEST_BIN](#macro_SETUP_PYTEST_BIN)
       - Macros [SETUP_RUN_PYTHON](#macro_SETUP_RUN_PYTHON) .. [SPLIT_FACTOR](#macro_SPLIT_FACTOR)
       - Macros [SRC](#macro_SRC) .. [SRC_C_AVX](#macro_SRC_C_AVX)
       - Macros [SRC_C_AVX2](#macro_SRC_C_AVX2) .. [TAG](#macro_TAG)
       - Macros [TASKLET](#macro_TASKLET) .. [UBERJAR_HIDE_EXCLUDE_PATTERN](#macro_UBERJAR_HIDE_EXCLUDE_PATTERN)
       - Macros [UBERJAR_HIDING_PREFIX](#macro_UBERJAR_HIDING_PREFIX) .. [USE_PYTHON3](#macro_USE_PYTHON3)
       - Macros [USE_RECIPE](#macro_USE_RECIPE) .. [YABS_SERVER_BUILD_YSON_INDEX](#macro_YABS_SERVER_BUILD_YSON_INDEX)
       - Macros [YABS_SERVER_PREPARE_QXL_FROM_SANDBOX](#macro_YABS_SERVER_PREPARE_QXL_FROM_SANDBOX) .. [YQL_ABI_VERSION](#macro_YQL_ABI_VERSION)
## Multimodules <a name="multimodules"></a>

###### Multimodule [DOCS][]() <a name="multimodule_DOCS"></a>
Definition of the documentation multimodule.

The output artifact is docs.tar.gz with statically generated site (using mkdocs as builder) when built directly
When PEERDIRed from other DOCS() behaves like a library (supplying own content and dependencies to build target).
Is only used together with the macros DOCS\_DIR(), DOCS\_CONFIG(), DOCS\_VARS().

###### Multimodule [PROTO\_LIBRARY][] <a name="multimodule_PROTO_LIBRARY"></a>
Definition of the project - proto library. More detailed documentation here.
https://wiki.yandex-team.ru/yatool/proto\_library/

Module declaration must be finished with the macro END().

###### Multimodule [PY23\_LIBRARY][]([name]) <a name="multimodule_PY23_LIBRARY"></a>
Build PY\_LIBRARY or PY3\_LIBRARY depending on incoming PEERDIR
Direct build or build by RECURSE creates both variants
For more information read https://wiki.yandex-team.ru/arcadia/python/pysrcs

Module declaration must be finished with the macro END().

###### Multimodule [PY23\_NATIVE\_LIBRARY][] <a name="multimodule_PY23_NATIVE_LIBRARY"></a>
Not documented yet.

###### Multimodule [SANDBOX\_TASK][] <a name="multimodule_SANDBOX_TASK"></a>
Not documented yet.

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

###### Module [GO\_BASE\_UNIT][]: \_BASE\_UNIT <a name="module_GO_BASE_UNIT"></a>
Not documented yet.

###### Module [GO\_LIBRARY][]: GO\_BASE\_UNIT <a name="module_GO_LIBRARY"></a>
Not documented yet.

###### Module [GO\_PROGRAM][]: GO\_BASE\_UNIT <a name="module_GO_PROGRAM"></a>
Not documented yet.

###### Module [GO\_TEST][]: GO\_PROGRAM <a name="module_GO_TEST"></a>
Not documented yet.

###### Module [GTEST][]([name]) <a name="module_GTEST"></a>
Definition of the test module based on gtest (contrib/libs/gtest contrib/libs/gmock)

Use publict documentation on gtest for details
Documentation about the Arcadia test system: https://wiki.yandex-team.ru/yatool/test/

###### Module [IOS\_APP][]: \_BASE\_UNIT <a name="module_IOS_APP"></a>
Not documented yet.

###### Module [JAVA\_LIBRARY][]() <a name="module_JAVA_LIBRARY"></a>
Documentation: https://wiki.yandex-team.ru/yatool/java/

###### Module [JAVA\_PLACEHOLDER][]: \_BASE\_UNIT <a name="module_JAVA_PLACEHOLDER"></a>
Not documented yet.

###### Module [JAVA\_PROGRAM][]() <a name="module_JAVA_PROGRAM"></a>
JAVA\_PROGRAM() - module for describing java programs.
Output artifacts: .jar and directory with all the jar to the classpath of the formation.

Documentation: https://wiki.yandex-team.ru/yatool/java/

###### Module [JTEST][]() <a name="module_JTEST"></a>
Definition of the project the Junit test.

Module to describe the java tests.
If requested, build system will scan the source code of the module for the presence of junit tests and run them.
Output artifacts: a jar, a directory of exhaust tests(if required run the tests) - test logs, system logs testiranja, temporary files, tests, etc.

Module declaration must be finished with the macro END().
Documentation: https://wiki.yandex-team.ru/yatool/test/#testynajava

###### Module [JTEST\_FOR][](x) <a name="module_JTEST_FOR"></a>
Module to describe the java tests.
In contrast to the JTEST , the build system will scan for the presence of the test sources of the module x . As x can be JAVA\_PROGRAM or JAVA\_LIBRARY . JTEST\_FOR also can have its own source, in this case they will be compiled and added to the classpath of a test run.
Output artifacts: a jar, a directory of exhaust tests(if requested tests are run).

Module declaration must be finished with the macro END().
Documentation: https://wiki.yandex-team.ru/yatool/test/#testynajava

###### Module [JUNIT5][]: JAVA\_PLACEHOLDER <a name="module_JUNIT5"></a>
Not documented yet.

###### Module [LIBRARY][]() <a name="module_LIBRARY"></a>
Definition of the static library module.
This is C++ library, and it selects peers from multimodules accordingly

It is recommended not to specify the name.

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

###### Module [PYMODULE][](name, major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_PYMODULE"></a>
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

###### Module [PY\_ANY\_MODULE][](name, major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_PY_ANY_MODULE"></a>
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

###### Module [R\_MODULE][](name, major\_ver [minor\_ver] [EXPORTS symlist\_file] [PREFIX prefix]) <a name="module_R_MODULE"></a>
Definition of the external module for R language
1. major\_ver and minor\_ver must be integers.
2. The resulting .so will have the prefix "lib".
3. Processing EXPORTS and PREFIX is the same as for DLL module

###### Module [TESTNG][]: JAVA\_PLACEHOLDER <a name="module_TESTNG"></a>
Not documented yet.

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

###### Macro [ADD\_COMPILABLE\_BYK][] <a name="macro_ADD_COMPILABLE_BYK"></a>
2 RUN\_PROGRAM under the condition

###### Macro [ADD\_COMPILABLE\_TRANSLATE][] <a name="macro_ADD_COMPILABLE_TRANSLATE"></a>
STRING(TOUPPER ...) + STRING(TOLOWER ...) + multiple RUN\_PROGRAM

###### Macro [ADD\_COMPILABLE\_TRANSLIT][] <a name="macro_ADD_COMPILABLE_TRANSLIT"></a>
STRING(TOUPPER ...) + STRING(TOLOWER ...) + multiple RUN\_PROGRAM

###### Macro [ADD\_DLLS\_TO\_JAR][]() <a name="macro_ADD_DLLS_TO_JAR"></a>
Not documented yet.

###### Macro [ADD\_PERL\_MODULE][](Dir, Module) <a name="macro_ADD_PERL_MODULE"></a>
Not documented yet.

###### Macro [ADD\_PYTEST\_BIN][] <a name="macro_ADD_PYTEST_BIN"></a>
Not documented yet.

###### Macro [ADD\_PYTEST\_SCRIPT][] <a name="macro_ADD_PYTEST_SCRIPT"></a>
Not documented yet.

###### Macro [ADD\_PY\_PROTO\_OUT][](Suf) <a name="macro_ADD_PY_PROTO_OUT"></a>
Not documented yet.

###### Macro [ADD\_TEST][] <a name="macro_ADD_TEST"></a>
Not documented yet.

###### Macro [ADD\_WAR][](Args...) <a name="macro_ADD_WAR"></a>
Not documented yet.

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

###### Macro [ASM\_PREINCLUDE][](PREINCLUDES...) <a name="macro_ASM_PREINCLUDE"></a>
Not documented yet.

###### Macro [BASE\_CODEGEN][](tool\_path prefix) <a name="macro_BASE_CODEGEN"></a>
Generator ${prefix}.cpp + ${prefix}.h files based on ${prefix}.in

###### Macro [BUILDWITH\_CYTHON\_C][] <a name="macro_BUILDWITH_CYTHON_C"></a>
Generates .c file from .pyx

###### Macro [BUILDWITH\_CYTHON\_CPP][] <a name="macro_BUILDWITH_CYTHON_CPP"></a>
Generates. cpp file from .pyx

###### Macro [BUILDWITH\_CYTHON\_CPP\_DEP][](Src, Dep, Options...) <a name="macro_BUILDWITH_CYTHON_CPP_DEP"></a>
Not documented yet.

###### Macro [BUILDWITH\_CYTHON\_C\_API\_H][] <a name="macro_BUILDWITH_CYTHON_C_API_H"></a>
BUILDWITH\_CYTHON\_C\_H with cdef api \_api.h file.

###### Macro [BUILDWITH\_CYTHON\_C\_DEP][](Src, Dep, Options...) <a name="macro_BUILDWITH_CYTHON_C_DEP"></a>
Not documented yet.

###### Macro [BUILDWITH\_CYTHON\_C\_H][] <a name="macro_BUILDWITH_CYTHON_C_H"></a>
BUILDWITH\_CYTHON\_C without .pyx infix and with cdef public .h file.

###### Macro [BUILDWITH\_RAGEL6][](Src, Options...) <a name="macro_BUILDWITH_RAGEL6"></a>
Not documented yet.

###### Macro BUILD\_CB(cbmodel cbname) <a name="macro_BUILD_CATBOOST"></a>
CatBoost specific macro
cbmodel - CatBoost model file name (\*.cmb)
cbname - name for a variable (of NCatboostCalcer::TCatboostCalcer type) to be available in CPP code

###### Macro [BUILD\_MN][]([CHECK] [PTR] [MULTI] mninfo mnname) <a name="macro_BUILD_MN"></a>
MatrixNet specific macro
Alternative macros:
1. BUILD\_MNS([CHECK] NAME listname mninfos...) - works faster and better for large files
2. BUILD\_MN\_ASM([CHECK] [PTR] [MULTI] mninfo mnname)

###### Macro [BUILD\_MNS][]([CHECK] NAME listname mninfos...) <a name="macro_BUILD_MNS"></a>
MatrixNet specific macro. Faster version of BUILD\_MN([CHECK] [PTR] [MULTI] mninfo mnname) for large files

###### Macro [BUILD\_MNS\_CPP][](NAME="", CHECK?, RANKING\_SUFFIX="", Files...) <a name="macro_BUILD_MNS_CPP"></a>
Not documented yet.

###### Macro [BUILD\_MNS\_FILE][](Input, Name, Output, Suffix, Check, Fml\_tool, AsmDataName) <a name="macro_BUILD_MNS_FILE"></a>
Not documented yet.

###### Macro [BUILD\_MNS\_FILES][] <a name="macro_BUILD_MNS_FILES"></a>
Not documented yet.

###### Macro [BUILD\_MNS\_HEADER][](NAME="", CHECK?, RANKING\_SUFFIX="", Files...) <a name="macro_BUILD_MNS_HEADER"></a>
Not documented yet.

###### Macro [BUILD\_ONLY\_IF][](variables)  _# builtin_ <a name="macro_BUILD_ONLY_IF"></a>
Print warning if all variables are false. For example, BUILD\_ONLY\_IF(LINUX WIN32)

###### Macro [BUILD\_PLNS][](Src...) <a name="macro_BUILD_PLNS"></a>
Not documented yet.

###### Macro [BUILTIN\_PYTHON][](script\_path args...) <a name="macro_BUILTIN_PYTHON"></a>
[CWD dir] [ENV key=value...]
[TOOL tools...] [IN inputs...]
[OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output])
[OUTPUT\_INCLUDES output\_includes...]

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

###### Macro [BUNDLE][] <a name="macro_BUNDLE"></a>
Not documented yet.

###### Macro [BUNDLE\_TARGET][](Target, Destination) <a name="macro_BUNDLE_TARGET"></a>
Not documented yet.

###### Macro [BYK\_NAME][] <a name="macro_BYK_NAME"></a>
Set 2 variables

###### Macro [CFG\_VARS][]() <a name="macro_CFG_VARS"></a>
Not documented yet.

###### Macro [CFLAGS][]([GLOBAL compiler\_flag]\* compiler\_flags) <a name="macro_CFLAGS"></a>
Add the specified flags to the compile line of files C/C++.
@params: GLOBAL - Distributes these flags on dependent projects
Note: remember about the incompatibility flags for gcc and cl (to set flags for the cl macro is used MSVC\_FLAGS).

###### Macro [CGO\_CFLAGS][](FLAGS...) <a name="macro_CGO_CFLAGS"></a>
Not documented yet.

###### Macro [CGO\_LDFLAGS][](FLAGS...) <a name="macro_CGO_LDFLAGS"></a>
Not documented yet.

###### Macro [CGO\_SRCS][](FILES...) <a name="macro_CGO_SRCS"></a>
Not documented yet.

###### Macro [CHECK\_CONFIG\_H][](Conf) <a name="macro_CHECK_CONFIG_H"></a>
Not documented yet.

###### Macro [CHECK\_DEPENDENT\_DIRS][](...)  _#bultin_ <a name="macro_CHECK_DEPENDENT_DIRS"></a>
Ignored

###### Macro [CHECK\_JAVA\_DEPS][](Arg) <a name="macro_CHECK_JAVA_DEPS"></a>
Not documented yet.

###### Macro [CLANG\_EMIT\_AST\_CXX][](Input, Output, Opts...) <a name="macro_CLANG_EMIT_AST_CXX"></a>
Not documented yet.

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
Add the specified flags to the compile line of files of type C.
@params: GLOBAL - Distributes these flags on dependent projects

###### Macro [COPY][] <a name="macro_COPY"></a>
Not documented yet.

###### Macro [COPY\_FILE][](File, Destination) <a name="macro_COPY_FILE"></a>
Not documented yet.

###### Macro [COPY\_FILE\_WITH\_DEPS][](File, Destination, OUTPUT\_INCLUDES[]) <a name="macro_COPY_FILE_WITH_DEPS"></a>
Not documented yet.

###### Macro [CPP\_EVLOG\_CMD][](File) <a name="macro_CPP_EVLOG_CMD"></a>
Not documented yet.

###### Macro [CPP\_PROTO\_CMD][](File) <a name="macro_CPP_PROTO_CMD"></a>
Not documented yet.

###### Macro [CPP\_PROTO\_EVLOG\_CMD][](File) <a name="macro_CPP_PROTO_EVLOG_CMD"></a>
Not documented yet.

###### Macro [CREATE\_BUILDINFO\_FOR][](GenHdr) <a name="macro_CREATE_BUILDINFO_FOR"></a>
Not documented yet.

###### Macro [CREATE\_GO\_SVNVERSION\_FOR][](GenHdr) <a name="macro_CREATE_GO_SVNVERSION_FOR"></a>
Not documented yet.

###### Macro [CREATE\_INIT\_PY][] <a name="macro_CREATE_INIT_PY"></a>
Not documented yet.

###### Macro [CREATE\_INIT\_PY\_FOR][] <a name="macro_CREATE_INIT_PY_FOR"></a>
Not documented yet.

###### Macro [CREATE\_INIT\_PY\_STRUCTURE][] <a name="macro_CREATE_INIT_PY_STRUCTURE"></a>
Not documented yet.

###### Macro [CREATE\_JAVA\_SVNVERSION\_FOR][](GenHdr) <a name="macro_CREATE_JAVA_SVNVERSION_FOR"></a>
Not documented yet.

###### Macro [CREATE\_SVNVERSION\_FOR][]() <a name="macro_CREATE_SVNVERSION_FOR"></a>
Creates a file with information about the svn revision and various other things

Can lead to frequent rebuilding programs.
Partly off by setting to yes NO\_SVN\_DEPENDS

###### Macro [CTEMPLATE\_VARNAMES][](File) <a name="macro_CTEMPLATE_VARNAMES"></a>
Not documented yet.

###### Macro [CUDA\_NVCC\_FLAGS][](compiler flags) <a name="macro_CUDA_NVCC_FLAGS"></a>
Add the specified flags to the compile line .cu-files.

###### Macro [CXXFLAGS][](compiler\_flags) <a name="macro_CXXFLAGS"></a>
Add the specified flags to the string compile-only C++files.

###### Macro [DARWIN\_SIGNED\_RESOURCE][](Resource, Relpath) <a name="macro_DARWIN_SIGNED_RESOURCE"></a>
Not documented yet.

###### Macro [DARWIN\_STRINGS\_RESOURCE][](Resource, Relpath) <a name="macro_DARWIN_STRINGS_RESOURCE"></a>
Not documented yet.

###### Macro [DATA][]([path...]) <a name="macro_DATA"></a>
Specifies the path to the data necessary test.
Valid: arcadia/<path> , arcadia\_tests\_data/<path>.

You can also specify the number of the resource required test in cbr format://<eng-id> (resource will be brought to the working directory of the test before it is started)
Used only inside a macro UNITTEST

Documentation: https://wiki.yandex-team.ru/yatool/test/#dannyeizrepozitorija

###### Macro [DEB\_VERSION][](File) <a name="macro_DEB_VERSION"></a>
Creates a header file DebianVersion.h define th DEBIAN\_VERSION taken from the File

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
Specify path to config .yml file for DOCS module if it differs from "%%project\_directory%%/mkdocs.yml".
Path must be either Arcadia root relative or lie in project directory.

###### Macro [DOCS\_DIR][](path) <a name="macro_DOCS_DIR"></a>
Specify directory with source .md files for DOCS module if it differs from project directory.
Path must be Arcadia root relative.

###### Macro [DOCS\_VARS][](variable1=value1 variable2=value2 ...) <a name="macro_DOCS_VARS"></a>
Specify a set of default values of template variables for DOCS module.
There must be no spaces around "=". Values will be treated as strings.

###### Macro [DUMPERF\_CODEGEN][](Prefix) <a name="macro_DUMPERF_CODEGEN"></a>
Not documented yet.

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

###### Macro [EXPORTS\_SCRIPT][](Arg) <a name="macro_EXPORTS_SCRIPT"></a>
Not documented yet.

###### Macro [EXPORT\_MAPKIT\_PROTO][]() <a name="macro_EXPORT_MAPKIT_PROTO"></a>
This macro is a temporary one and should be changed to EXPORT\_YMAPS\_PROTO
when transition of mapsmobi to arcadia is finished

###### Macro [EXPORT\_YMAPS\_PROTO][]() <a name="macro_EXPORT_YMAPS_PROTO"></a>
Not documented yet.

###### Macro [EXTERNAL\_JAR][](library.jar) <a name="macro_EXTERNAL_JAR"></a>
Is used to indicate the collected external java libraries inside the project JAVA\_LIBRARY()

Documentation: https://wiki.yandex-team.ru/yatool/java/#ispolzovanievneshnixmavenbibliotek

###### Macro [EXTERNAL\_RESOURCE][](...)  _#builtin deprecated_ <a name="macro_EXTERNAL_RESOURCE"></a>
Deprecated

###### Macro [EXTRADIR][](...)  _#bultin_ <a name="macro_EXTRADIR"></a>
Ignored

###### Macro [EXTRALIBS][](liblist)  _# builtin_ <a name="macro_EXTRALIBS"></a>
Add external dynamic libraries during program linkage stage

###### Macro [EXTRALIBS\_STATIC][](liblist) <a name="macro_EXTRALIBS_STATIC"></a>
Add the specified external static libraries in linking the program

###### Macro [FAT\_RESOURCE][] <a name="macro_FAT_RESOURCE"></a>
Not documented yet.

###### Macro [FILES][] <a name="macro_FILES"></a>
Not documented yet.

###### Macro [FLAT\_JOIN\_SRCS\_GLOBAL][](Out, Src...) <a name="macro_FLAT_JOIN_SRCS_GLOBAL"></a>
Not documented yet.

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

###### Macro [GENERATE\_ENUM\_SERIALIZATION][](headername.h) <a name="macro_GENERATE_ENUM_SERIALIZATION"></a>
Build funkciigi input-output of enumeration members defined in the header
Documentation: https://wiki.yandex-team.ru/yatool/HowToWriteYaMakeFiles/

###### Macro [GENERATE\_ENUM\_SERIALIZATION\_WITH\_HEADER][](File) <a name="macro_GENERATE_ENUM_SERIALIZATION_WITH_HEADER"></a>
Not documented yet.

###### Macro [GENERATE\_PY\_EVS][](FILES...) <a name="macro_GENERATE_PY_EVS"></a>
Not documented yet.

###### Macro [GENERATE\_PY\_PROTOS][] <a name="macro_GENERATE_PY_PROTOS"></a>
Macro is obsolete and not recommended for use! Generate python bindings for protobuf files.

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

###### Macro [GO\_ASM\_FLAGS][](flags) <a name="macro_GO_ASM_FLAGS"></a>
Add the specified flags to the go asm compile icommand line.

###### Macro [GO\_CGO1\_FLAGS][](flags) <a name="macro_GO_CGO1_FLAGS"></a>
Add the specified flags to the go cgo compile command line.

###### Macro [GO\_CGO2\_FLAGS][](flags) <a name="macro_GO_CGO2_FLAGS"></a>
Add the specified flags to the go cgo compile command line.

###### Macro [GO\_COMPILE\_CGO1][](NAME, FLAGS[], FILES...) <a name="macro_GO_COMPILE_CGO1"></a>
Not documented yet.

###### Macro [GO\_COMPILE\_CGO2][](NAME, C\_FILES[], S\_FILES[], FILES...) <a name="macro_GO_COMPILE_CGO2"></a>
Not documented yet.

###### Macro [GO\_COMPILE\_FLAGS][](flags) <a name="macro_GO_COMPILE_FLAGS"></a>
Add the specified flags to the go compile command line.

###### Macro [GO\_COMPILE\_SYMABIS][](ASM\_FILES...) <a name="macro_GO_COMPILE_SYMABIS"></a>
Not documented yet.

###### Macro [GO\_FAKE\_OUTPUT][](FILES...) <a name="macro_GO_FAKE_OUTPUT"></a>
Not documented yet.

###### Macro [GO\_GRPC][]() <a name="macro_GO_GRPC"></a>
Not documented yet.

###### Macro [GO\_LDFLAGS][](FLAGS...) <a name="macro_GO_LDFLAGS"></a>
Not documented yet.

###### Macro [GO\_LINK\_EXE\_IMPL][](GO\_FILES...) <a name="macro_GO_LINK_EXE_IMPL"></a>
Not documented yet.

###### Macro [GO\_LINK\_FLAGS][](flags) <a name="macro_GO_LINK_FLAGS"></a>
Add the specified flags to the go link command line.

###### Macro [GO\_LINK\_LIB\_IMPL][](GO\_FILES...) <a name="macro_GO_LINK_LIB_IMPL"></a>
Not documented yet.

###### Macro [GO\_LINK\_TEST\_IMPL][](GO\_TEST\_FILES[], GO\_XTEST\_FILES[], GO\_FILES...) <a name="macro_GO_LINK_TEST_IMPL"></a>
Not documented yet.

###### Macro [GO\_PACKAGE\_NAME][](NAME) <a name="macro_GO_PACKAGE_NAME"></a>
Not documented yet.

###### Macro [GO\_PROTOC\_PLUGIN\_ARGS\_BASE][](Name, Tool, Plugins="") <a name="macro_GO_PROTOC_PLUGIN_ARGS_BASE"></a>
Not documented yet.

###### Macro [GO\_PROTO\_CMD][](File) <a name="macro_GO_PROTO_CMD"></a>
Not documented yet.

###### Macro [GO\_SRCS][](FILES...) <a name="macro_GO_SRCS"></a>
Not documented yet.

###### Macro [GO\_TEST\_FOR][](path/to/module)  _#builtin_ <a name="macro_GO_TEST_FOR"></a>
Produces go test for specified module

###### Macro [GO\_TEST\_SRCS][](FILES...) <a name="macro_GO_TEST_SRCS"></a>
Not documented yet.

###### Macro [GO\_UNUSED\_SRCS][](FLAGS...) <a name="macro_GO_UNUSED_SRCS"></a>
Not documented yet.

###### Macro [GO\_XTEST\_SRCS][](FILES...) <a name="macro_GO_XTEST_SRCS"></a>
Not documented yet.

###### Macro [GO\_YDB\_GRPC][](Plugins...) <a name="macro_GO_YDB_GRPC"></a>
Not documented yet.

###### Macro [GRPC][]() <a name="macro_GRPC"></a>
Not documented yet.

###### Macro IDEA\_RESOURCE\_DIRS(<additional dirs>) <a name="macro_IDEA_EXCLUDE_DIRS"></a>
Macro sets specified resource directories in an idea project generated by ya ide idea

###### Macro [IDEA\_RESOURCE\_DIRS][](Args...) <a name="macro_IDEA_RESOURCE_DIRS"></a>
Not documented yet.

###### Macro [IF][](expression) .. ELSEIF(expression) .. ELSE() .. ENDIF()  _#builtin_ <a name="macro_IF"></a>
Not documented yet.

###### Macro [IMPORT\_YMAPS\_PROTO][]() <a name="macro_IMPORT_YMAPS_PROTO"></a>
Not documented yet.

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
Not documented yet.

###### Macro [JAVA\_EVLOG\_CMD][](File) <a name="macro_JAVA_EVLOG_CMD"></a>
Not documented yet.

###### Macro [JAVA\_IGNORE\_CLASSPATH\_CLASH\_FOR][]([classes]) <a name="macro_JAVA_IGNORE_CLASSPATH_CLASH_FOR"></a>
Ignore classpath clash test fails for classes

###### Macro [JAVA\_MODULE][] <a name="macro_JAVA_MODULE"></a>
Not documented yet.

###### Macro [JAVA\_PROTO\_CMD][](File) <a name="macro_JAVA_PROTO_CMD"></a>
Not documented yet.

###### Macro [JAVA\_SRCS][](srcs) <a name="macro_JAVA_SRCS"></a>
Using the macro JAVA\_SRCS() specifies the java source files and resources. A macro can be contained in any of four java modules.
Keywords:
1. X SRCDIR - specify the directory x is performed relatively to search the source code for these patterns. If there is no SRCDIR, the source will be searched relative to the module directory.
2. PACKAGE\_PREFIX x - use if source paths relative to the SRCDIR does not coincide with the full class names. For example, if all sources of module are in the same package, you can create a directory package/name , and just put the source code in the SRCDIR and specify PACKAGE\_PREFIX package.name.

@example:
Пример
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

###### Macro [JOIN\_SRCS][](Out, Src...) <a name="macro_JOIN_SRCS"></a>
Not documented yet.

###### Macro [JOIN\_SRCS\_GLOBAL][](Out, Src...) <a name="macro_JOIN_SRCS_GLOBAL"></a>
Not documented yet.

###### Macro [JVM\_ARGS][](Args...) <a name="macro_JVM_ARGS"></a>
Not documented yet.

###### Macro [LDFLAGS][](linker\_flags) <a name="macro_LDFLAGS"></a>
Add flags to the linkage command line of executable or shared library/dll.
Note: remember about the incompatibility of flags for gcc and cl.

###### Macro [LDFLAGS\_FIXED][](Flags...) <a name="macro_LDFLAGS_FIXED"></a>
Not documented yet.

###### Macro [LICENSE][](licenses...) <a name="macro_LICENSE"></a>
Specify the license of the module, separated by spaces

A license must be prescribed for contribs

###### Macro [LINT][](<none|base|strict>) <a name="macro_LINT"></a>
Set linting levem for sources of the module

###### Macro [LJ\_21\_ARCHIVE][] <a name="macro_LJ_21_ARCHIVE"></a>
Not documented yet.

###### Macro [LJ\_ARCHIVE][] <a name="macro_LJ_ARCHIVE"></a>
Not documented yet.

###### Macro [LLVM\_BC][] <a name="macro_LLVM_BC"></a>
Not documented yet.

###### Macro [LLVM\_COMPILE\_C][](Input, Output, Opts...) <a name="macro_LLVM_COMPILE_C"></a>
Not documented yet.

###### Macro [LLVM\_COMPILE\_CXX][](Input, Output, Opts...) <a name="macro_LLVM_COMPILE_CXX"></a>
Not documented yet.

###### Macro [LLVM\_COMPILE\_LL][](Input, Output, Opts...) <a name="macro_LLVM_COMPILE_LL"></a>
Not documented yet.

###### Macro [LLVM\_LINK][](Output, Inputs...) <a name="macro_LLVM_LINK"></a>
Not documented yet.

###### Macro [LLVM\_OPT][](Input, Output, Opts...) <a name="macro_LLVM_OPT"></a>
Not documented yet.

###### Macro [LSAN\_SUPPRESSIONS][] <a name="macro_LSAN_SUPPRESSIONS"></a>
LSAN\_SUPPRESSIONS() - allows to specify files with suppression notation which will be used by default.
See https://clang.llvm.org/docs/AddressSanitizer.html#suppressing-memory-leaks
for details.

###### Macro [LUA][](script\_path args... <a name="macro_LUA"></a>
[CWD dir] [ENV key=value...]
[TOOL tools...] [IN inputs...]
[OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output])
[OUTPUT\_INCLUDES output\_includes...]

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

###### Macro [LUAJIT\_21\_OBJDUMP][](Src, OUT="") <a name="macro_LUAJIT_21_OBJDUMP"></a>
Not documented yet.

###### Macro [LUAJIT\_OBJDUMP][](Src, OUT="") <a name="macro_LUAJIT_OBJDUMP"></a>
Not documented yet.

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

###### Macro [MAPKIT\_CPP\_PROTO\_CMD][](File) <a name="macro_MAPKIT_CPP_PROTO_CMD"></a>
Not documented yet.

###### Macro [MAPKIT\_ENABLE\_WHOLE\_ARCHIVE][]() <a name="macro_MAPKIT_ENABLE_WHOLE_ARCHIVE"></a>
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
Add the specified flags to the compile line .masm files.

###### Macro [MAVEN\_GROUP\_ID][](group\_id\_for\_maven\_export) <a name="macro_MAVEN_GROUP_ID"></a>
Set maven export group id for JAVA\_PROGRAM() and JAVA\_LIBRARY()

###### Macro [MESSAGE][]([severity] message)  _# builtin_ <a name="macro_MESSAGE"></a>
Print message with given severity level (STATUS, FATAL\_ERROR)

###### Macro [METAQUERYFILES][](filenames...) _#deprecated_ <a name="macro_METAQUERYFILES"></a>
Deprecated

###### Macro [MSVC\_FLAGS][]([GLOBAL compiler\_flag]\* compiler\_flags) <a name="macro_MSVC_FLAGS"></a>
Add the specified flags to the compilation line of C/C++files.
Flags apply only if the compiler used is MSVC (cl.exe)

###### Macro [MX\_BIN\_TO\_INFO][](Src) <a name="macro_MX_BIN_TO_INFO"></a>
Not documented yet.

###### Macro [MX\_FORMULAS][] <a name="macro_MX_FORMULAS"></a>
Not documented yet.

###### Macro [MX\_GEN\_TABLE][](Srcs...) <a name="macro_MX_GEN_TABLE"></a>
Not documented yet.

###### Macro [NEED\_CHECK][] <a name="macro_NEED_CHECK"></a>
Commits to the project marked with this macro will be blocked by pre-commit check and then will be
automatically merged to trunk only if there is no new broken build targets in check results.
The use of this macro is disabled by default.

###### Macro [NEED\_REVIEW][] <a name="macro_NEED_REVIEW"></a>
Mark the project as needing review.
Reviewers are listed in the macro OWNER. The use of this macro is disabled by default.
Details can be found here: https://clubs.at.yandex-team.ru/arcadia/6104

###### Macro [NO\_BUILD\_IF][](variables)  _# builtin_ <a name="macro_NO_BUILD_IF"></a>
Print warning if some variable is true

###### Macro [NO\_CHECK\_IMPORTS][]([patterns]) <a name="macro_NO_CHECK_IMPORTS"></a>
Do not run checks on imports of Python modules.
Optional parameter mask patterns describes the names of the modules that do not need to check.

###### Macro [NO\_CLANG\_COVERAGE][]() <a name="macro_NO_CLANG_COVERAGE"></a>
Not documented yet.

###### Macro [NO\_CODENAVIGATION][]() <a name="macro_NO_CODENAVIGATION"></a>
Not documented yet.

###### Macro [NO\_COMPILER\_WARNINGS][]() <a name="macro_NO_COMPILER_WARNINGS"></a>
Not documented yet.

###### Macro [NO\_DEBUG\_INFO][]() <a name="macro_NO_DEBUG_INFO"></a>
Not documented yet.

###### Macro [NO\_GPL][] <a name="macro_NO_GPL"></a>
To check the license of the dependencies, indicated by macro LICENSE, and fail the build in case of detection of GPL-like licenses.
Do not use with LIBRARY().
Issues a warning if the detected dependencies from contrib-ov in which the license is not specified.

###### Macro [NO\_JOIN\_SRC][]() <a name="macro_NO_JOIN_SRC"></a>
Not documented yet.

###### Macro [NO\_LIBC][]() <a name="macro_NO_LIBC"></a>
NO\_RUNTIME() + DISABLE(MUSL)

###### Macro [NO\_LINT][]() <a name="macro_NO_LINT"></a>
Do not check for style files included in PY\_SRCS, TEST\_SRCS, JAVA\_SRCS

###### Macro [NO\_NEED\_CHECK][] <a name="macro_NO_NEED_CHECK"></a>
Commits to the project marked with this macro will not be affected by higher-level NEED\_CHECK macro.

###### Macro [NO\_OPTIMIZE][]() <a name="macro_NO_OPTIMIZE"></a>
Not documented yet.

###### Macro [NO\_OPTIMIZE\_PY\_PROTOS][]() <a name="macro_NO_OPTIMIZE_PY_PROTOS"></a>
Not documented yet.

###### Macro [NO\_PLATFORM][] <a name="macro_NO_PLATFORM"></a>
NO\_LIBC() + ENABLE(NOPLATFORM)

###### Macro [NO\_PLATFORM\_RESOURCES][]() <a name="macro_NO_PLATFORM_RESOURCES"></a>
Not documented yet.

###### Macro [NO\_PYTHON\_INCLUDES][]() <a name="macro_NO_PYTHON_INCLUDES"></a>
Not documented yet.

###### Macro [NO\_RUNTIME][]() <a name="macro_NO_RUNTIME"></a>
This macro:
1. Sets the ENABLE(NOUTIL) + DISABLE(USE\_INTERNAL\_STL)
2. If the project that contains the macro NO\_RUNTIME(), peerdir-it project does not contain NO\_RUNTIME() => Warning

###### Macro [NO\_SANITIZE][]() <a name="macro_NO_SANITIZE"></a>
DISABLE(SANITIZER\_TYPE)

###### Macro [NO\_SANITIZE\_COVERAGE][]() <a name="macro_NO_SANITIZE_COVERAGE"></a>
Not documented yet.

###### Macro [NO\_SSE4][]() <a name="macro_NO_SSE4"></a>
Compile module without SSE4

###### Macro [NO\_UTIL][]() <a name="macro_NO_UTIL"></a>
Not documented yet.

###### Macro [NO\_WERROR][]() <a name="macro_NO_WERROR"></a>
Back-to-WERROR()
Priorities: NO\_COMPILER\_WARNINGS > NO\_WERROR > WERROR\_MODE > WERROR

###### Macro [NO\_WSHADOW][]() <a name="macro_NO_WSHADOW"></a>
Not documented yet.

###### Macro [ONLY\_TAGS][](tags...)  _# builtin_ <a name="macro_ONLY_TAGS"></a>
Instantiate from multimodule only variants with tags listed

###### Macro [OPTIMIZE\_PY\_PROTOS][]() <a name="macro_OPTIMIZE_PY_PROTOS"></a>
Not documented yet.

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

###### Macro [PROTO2FBS][](File) <a name="macro_PROTO2FBS"></a>
Not documented yet.

###### Macro [PROTO\_PLUGIN\_ARGS\_BASE][](Name, Tool) <a name="macro_PROTO_PLUGIN_ARGS_BASE"></a>
Not documented yet.

###### Macro [PROVIDES][](Name...) <a name="macro_PROVIDES"></a>
Specifies provided features. The names must be C identifiers. Different
libraries providing the same features can not be linked into one program.

###### Macro [PY3\_COMPILE\_BYTECODE][](SrcX, Src) <a name="macro_PY3_COMPILE_BYTECODE"></a>
Not documented yet.

###### Macro [PYTHON][](script\_path args... <a name="macro_PYTHON"></a>
[CWD dir] [ENV key=value...]
[TOOL tools...] [IN inputs...]
[OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output])
[OUTPUT\_INCLUDES output\_includes...]

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
This macro adds include path for Python headers (Python3 variant)
This should be used in 2 cases only:
- In PYMODULE since it compiles into .so and uses external Python runtime
- In system Python libraries themselves since peerdir there may create a loop
Never use this macro in PY3\_PROGRAM and PY3\_LIBRARY and PY23\_LIBRARY: they have everything by default
In all other cases use USE\_PYTHON3() macro instead
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs

###### Macro PYTHON2\_MODULE() <a name="macro_PYTHON3_MODULE"></a>
Use in PY\_ANY\_MODULE to set it up for Python 3.x

###### Macro [PYTHON\_ADDINCL][]() <a name="macro_PYTHON_ADDINCL"></a>
This macro adds include path for Python headers (Python2 variant)
This should be used in 2 cases only:
- In PYMODULE since it compiles into .so and uses external Python runtime
- In system Python libraries themselves since peerdir there may create a loop
Never use this macro in PY\_PROGRAM, PY\_LIBRARY and PY23\_LIBRARY: they have everything by default
In all other cases use USE\_PYTHON macro instead
Documentation: https://wiki.yandex-team.ru/devtools/commandsandvars/py\_srcs

###### Macro [PYTHON\_PATH][](Path) <a name="macro_PYTHON_PATH"></a>
Not documented yet.

###### Macro [PY\_CODENAV][](For) <a name="macro_PY_CODENAV"></a>
Not documented yet.

###### Macro [PY\_COMPILE\_BYTECODE][](SrcX, Src) <a name="macro_PY_COMPILE_BYTECODE"></a>
Not documented yet.

###### Macro PY\_DOCTEST(Packages...) <a name="macro_PY_DOCTESTS"></a>
Add to the test doctests for specified Python packages
The packages should be part of a test (listed as sources of the test or its PEERDIRs).

###### Macro [PY\_EVLOG\_CMD][](File) <a name="macro_PY_EVLOG_CMD"></a>
Not documented yet.

###### Macro [PY\_EVLOG\_CMD\_BASE][](File, Suf, Args...) <a name="macro_PY_EVLOG_CMD_BASE"></a>
Not documented yet.

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

###### Macro [PY\_PROTO\_CMD][](File) <a name="macro_PY_PROTO_CMD"></a>
Not documented yet.

###### Macro [PY\_PROTO\_CMD\_BASE][](File, Suf, Args...) <a name="macro_PY_PROTO_CMD_BASE"></a>
Not documented yet.

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

###### Macro [RUN\_PROGRAM][](tool\_path args... <a name="macro_RUN_PROGRAM"></a>
[CWD dir] [ENV key=value...]
[TOOL tools...] [IN inputs...]
[OUT[\_NOAUTO] outputs...] [STDOUT[\_NOAUTO] output])
[OUTPUT\_INCLUDES output\_includes...]

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
Not documented yet.

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

###### Macro [SPLIT\_CODEGEN][] <a name="macro_SPLIT_CODEGEN"></a>
Not documented yet.

###### Macro SPLIT\_CODEGEN(tool prefix opts... [OUT\_NUM num] [OUTPUT\_INCLUDES output\_includes...]) <a name="macro_SPLIT_CODEGEN_BASE"></a>
Generator of a certain number .the. cpp file + one header .h file from .in

Supports keywords:
1. OUT\_NUM <the number of generated Prefix.N.cpp default 25 (N varies from 0 to 24)>
2. OUTPUT\_INCLUDES <path to files that will be included in generalnyj of macro files>

###### Macro [SPLIT\_DWARF][]() <a name="macro_SPLIT_DWARF"></a>
Not documented yet.

###### Macro [SPLIT\_FACTOR][](x) <a name="macro_SPLIT_FACTOR"></a>
Sets the number of chunks for parallel run tests when using FORK\_TESTS, FORK\_SUBTESTS
As yourself use means the macro FORK\_TESTS.

Supports C++ ut and PyTest.

Documentation about the system test: https://wiki.yandex-team.ru/yatool/test/

###### Macro [SRC][](FILE, FLAGS...) <a name="macro_SRC"></a>
Not documented yet.

###### Macro [SRCDIR][](dirlist)  _# builtin_ <a name="macro_SRCDIR"></a>
Add the specified directories to the list of those in which the source files will be searched
Available only for arcadia/contrib

###### Macro [SRCS][]([GLOBAL file]\* filenames...) <a name="macro_SRCS"></a>
@brief Source files of the project
Arcadia Paths from the root and is relative to the project's LIST are supported

@param GLOBAL marks next file as direct input link of the program/shared library project built into
The scope of the GLOBAL keyword - the following file (that is, in the case of SRCS(GLOBAL foo.the cpp bar.cpp) global will be only foo.cpp)

@example:
Consider the file to ya.make:

    LIBRARY(test\_global)
    SRCS(GLOBAL foo.cpp)
    END()

SRCS(GLOBAL foo.cpp) - use to PROGRAM and DLL that is implicitly\* peerdir Yat test\_global had clearly been linked with foo.cpp.o.
- we say that prog1 (or dll1) implicitly peerdirs libN, if there is a
relationship of the form: "prog1 (or dll1) peerdirs lib1, lib1 peerdirs lib2,
..., libN-1 peerdirs libN"

###### Macro [SRC\_CPP\_AVX][](FILE, FLAGS...) <a name="macro_SRC_CPP_AVX"></a>
Not documented yet.

###### Macro [SRC\_CPP\_AVX2][](FILE, FLAGS...) <a name="macro_SRC_CPP_AVX2"></a>
Not documented yet.

###### Macro [SRC\_CPP\_SSE2][](FILE, FLAGS...) <a name="macro_SRC_CPP_SSE2"></a>
Not documented yet.

###### Macro [SRC\_CPP\_SSE3][](FILE, FLAGS...) <a name="macro_SRC_CPP_SSE3"></a>
Not documented yet.

###### Macro [SRC\_CPP\_SSE41][](FILE, FLAGS...) <a name="macro_SRC_CPP_SSE41"></a>
Not documented yet.

###### Macro [SRC\_CPP\_SSSE3][](FILE, FLAGS...) <a name="macro_SRC_CPP_SSSE3"></a>
Not documented yet.

###### Macro [SRC\_C\_AVX][](FILE, FLAGS...) <a name="macro_SRC_C_AVX"></a>
Not documented yet.

###### Macro [SRC\_C\_AVX2][](FILE, FLAGS...) <a name="macro_SRC_C_AVX2"></a>
Not documented yet.

###### Macro [SRC\_C\_SSE2][](FILE, FLAGS...) <a name="macro_SRC_C_SSE2"></a>
Not documented yet.

###### Macro [SRC\_C\_SSE3][](FILE, FLAGS...) <a name="macro_SRC_C_SSE3"></a>
Not documented yet.

###### Macro [SRC\_C\_SSE41][](FILE, FLAGS...) <a name="macro_SRC_C_SSE41"></a>
Not documented yet.

###### Macro [SRC\_C\_SSSE3][](FILE, FLAGS...) <a name="macro_SRC_C_SSSE3"></a>
Not documented yet.

###### Macro [STRIP][]() <a name="macro_STRIP"></a>
Not documented yet.

###### Macro [STRUCT\_CODEGEN][](prefix cpp\_dependant) <a name="macro_STRUCT_CODEGEN"></a>
A special case BASE\_CODEGEN , in which the tool is used kernel/struct\_codegen/codegen\_tool

###### Macro [SYMLINK][](from to) <a name="macro_SYMLINK"></a>
Add symlink

###### Macro [SYSTEM\_PROPERTIES][](Args...) <a name="macro_SYSTEM_PROPERTIES"></a>
Not documented yet.

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

Is only used inside of the macro TEST

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
UBERJAR is a single all-in-one jar-archive includes all needed him (achievable PERDIR) Java dependence. Also supported pros classes inside the archive to a different package (similar to the Maven-shade-plugin).
In order to collect uberjar, you need to add uberjar macros to the JAVA\_PROGRAM module .
Settings
You can use the following macros to configure the saudi archive:
1. UBERJAR\_HIDING\_PREFIX prefix for roaming (shadows) classes (default still remains the same)
2. UBERJAR\_HIDE\_EXCLUDE\_PATTERN for template classes that permission is not necessary(by default, all classes are moved )
3. UBERJAR\_PATH\_EXCLUDE\_PREFIX the prefix for paths that should not get in the jar archive

###### Macro [UBERJAR\_HIDE\_EXCLUDE\_PATTERN][](Args...) <a name="macro_UBERJAR_HIDE_EXCLUDE_PATTERN"></a>
Not documented yet.

###### Macro [UBERJAR\_HIDING\_PREFIX][](Arg) <a name="macro_UBERJAR_HIDING_PREFIX"></a>
Not documented yet.

###### Macro [UBERJAR\_PATH\_EXCLUDE\_PREFIX][](Args...) <a name="macro_UBERJAR_PATH_EXCLUDE_PREFIX"></a>
Not documented yet.

###### Macro [USE\_ERROR\_PRONE][]() <a name="macro_USE_ERROR_PRONE"></a>
Not documented yet.

###### Macro [USE\_LINKER][]() <a name="macro_USE_LINKER"></a>
Not documented yet.

###### Macro [USE\_LINKER\_BFD][]() <a name="macro_USE_LINKER_BFD"></a>
Not documented yet.

###### Macro [USE\_LINKER\_GOLD][]() <a name="macro_USE_LINKER_GOLD"></a>
Not documented yet.

###### Macro [USE\_LINKER\_LLD][]() <a name="macro_USE_LINKER_LLD"></a>
Not documented yet.

###### Macro [USE\_PERL\_LIB][]() <a name="macro_USE_PERL_LIB"></a>
Not documented yet.

###### Macro [USE\_PYTHON][]() <a name="macro_USE_PYTHON"></a>
This adds Python2 library to your LIBRARY and makes it Python2-compatible
If you'd like to use #include <Python.h> with Python2 specify USE\_PYTHON
If you'd like to use #include <Python.h> with both Python2 and Python3 convert your LIBRARY to PY23\_LIBRARY

###### Macro [USE\_PYTHON3][]() <a name="macro_USE_PYTHON3"></a>
This adds Python3 library to your LIBRARY and makes it Python3-compatible
If you'd like to use #include <Python.h> with Python3 specify USE\_PYTHON3
If you'd like to use #include <Python.h> with both Python2 and Python3 convert your LIBRARY to PY23\_LIBRARY

###### Macro [USE\_RECIPE][](path [arg1 arg2...]) <a name="macro_USE_RECIPE"></a>
Provides prepared environment via recipe for test
Documentation: https://wiki.yandex-team.ru/yatool/test/recipes

###### Macro [USE\_YQL][]() <a name="macro_USE_YQL"></a>
Add required PEERDIRs for YQL module.
Can be used for in-place module definition in user LIBRARY.

###### Macro [VERSION][](Flags...) <a name="macro_VERSION"></a>
Not documented yet.

###### Macro [WERROR][]() <a name="macro_WERROR"></a>
Consider warning for error in the current project.
In the bright future will be removed, since WERROR is the default.
Priorities: NO\_COMPILER\_WARNINGS > NO\_WERROR > WERROR\_MODE > WERROR

###### Macro [WITH\_JDK][]() <a name="macro_WITH_JDK"></a>
Not documented yet.

###### Macro [XS\_PROTO][](File, Dir, Outputs...) <a name="macro_XS_PROTO"></a>
Not documented yet.

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

###### Macro [YDB\_GRPC][]() <a name="macro_YDB_GRPC"></a>
Not documented yet.

###### Macro [YMAPS\_GENERATE\_SPROTO\_HEADER][](File) <a name="macro_YMAPS_GENERATE_SPROTO_HEADER"></a>
Not documented yet.

###### Macro [YMAPS\_SPROTO][](FILES...) <a name="macro_YMAPS_SPROTO"></a>
Not documented yet.

###### Macro [YQL\_ABI\_VERSION][](major minor release)) <a name="macro_YQL_ABI_VERSION"></a>
Specifying the supported ABI for YQL\_UDF.

 [DOCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1844
 [PROTO\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3881
 [PY23\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4267
 [PY23\_NATIVE\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4282
 [SANDBOX\_TASK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2322
 [YQL\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1696
 [BENCHMARK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1249
 [BOOSTTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L988
 [DEV\_DLL\_PROXY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1552
 [DLL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1530
 [DLL\_JAVA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1577
 [EXECTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1237
 [FAT\_OBJECT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1368
 [FUZZ]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L963
 [GO\_BASE\_UNIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4145
 [GO\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4200
 [GO\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4209
 [GO\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4236
 [GTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1188
 [IOS\_APP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4335
 [JAVA\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2273
 [JAVA\_PLACEHOLDER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2246
 [JAVA\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2267
 [JTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2286
 [JTEST\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2315
 [JUNIT5]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2301
 [LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1329
 [METAQUERY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2055
 [PACKAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1787
 [PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L871
 [PY3TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1176
 [PY3TEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1160
 [PY3\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1966
 [PY3\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L891
 [PYMODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1475
 [PYTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1154
 [PYTEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1140
 [PY\_ANY\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1435
 [PY\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1942
 [PY\_PACKAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1831
 [PY\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L880
 [RESOURCES\_LIBRARY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1346
 [R\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1507
 [TESTNG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2295
 [UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1598
 [UDF\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1585
 [UDF\_LIB]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1606
 [UNION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1820
 [UNITTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L936
 [UNITTEST\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1271
 [UNITTEST\_WITH\_CUSTOM\_ENTRY\_POINT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L950
 [YCR\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L910
 [YQL\_PYTHON3\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1738
 [YQL\_PYTHON\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1717
 [YQL\_PYTHON\_UDF\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1759
 [YQL\_UDF\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1621
 [YQL\_UDF\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1201
 [YT\_UNITTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L943
 [ACCELEO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/scarab_cant_clash.py?rev=5002020#L4
 [ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [ADDINCLSELF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/addinclself.py?rev=5002020#L2
 [ADD\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L411
 [ADD\_CHECK\_PY\_IMPORTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L468
 [ADD\_COMPILABLE\_BYK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2531
 [ADD\_COMPILABLE\_TRANSLATE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2544
 [ADD\_COMPILABLE\_TRANSLIT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2551
 [ADD\_DLLS\_TO\_JAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2201
 [ADD\_PERL\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2480
 [ADD\_PYTEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L530
 [ADD\_PYTEST\_SCRIPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L504
 [ADD\_PY\_PROTO\_OUT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L335
 [ADD\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L367
 [ADD\_WAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2123
 [ADD\_YTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L309
 [ALLOCATOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2470
 [ALL\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1680
 [ANNOTATION\_PROCESSOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2156
 [ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3092
 [ARCHIVE\_ASM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3073
 [ARCHIVE\_BY\_KEYS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3096
 [ASM\_PREINCLUDE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3798
 [BASE\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3155
 [BUILDWITH\_CYTHON\_C]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3015
 [BUILDWITH\_CYTHON\_CPP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3002
 [BUILDWITH\_CYTHON\_CPP\_DEP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3007
 [BUILDWITH\_CYTHON\_C\_API\_H]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3036
 [BUILDWITH\_CYTHON\_C\_DEP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3020
 [BUILDWITH\_CYTHON\_C\_H]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3028
 [BUILDWITH\_RAGEL6]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3041
 [BUILD\_CATBOOST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3398
 [BUILD\_MN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3388
 [BUILD\_MNS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3422
 [BUILD\_MNS\_CPP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3409
 [BUILD\_MNS\_FILE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3403
 [BUILD\_MNS\_FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/build_mn_files.py?rev=5002020#L4
 [BUILD\_MNS\_HEADER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3413
 [BUILD\_ONLY\_IF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [BUILD\_PLNS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3469
 [BUILTIN\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3617
 [BUNDLE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/bundle.py?rev=5002020#L4
 [BUNDLE\_TARGET]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2598
 [BYK\_NAME]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2520
 [CFG\_VARS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3148
 [CFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3206
 [CGO\_CFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4116
 [CGO\_LDFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4122
 [CGO\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4105
 [CHECK\_CONFIG\_H]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L294
 [CHECK\_DEPENDENT\_DIRS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [CHECK\_JAVA\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2074
 [CLANG\_EMIT\_AST\_CXX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3691
 [COMPILE\_C\_AS\_CXX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3651
 [COMPILE\_SWIFT\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4371
 [CONDITIONAL\_PEERDIR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/conditional_peerdir.py?rev=5002020#L4
 [CONFIGURE\_FILE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3144
 [CONLYFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3219
 [COPY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/cp.py?rev=5002020#L6
 [COPY\_FILE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2590
 [COPY\_FILE\_WITH\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2594
 [CPP\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L372
 [CPP\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L367
 [CPP\_PROTO\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L377
 [CREATE\_BUILDINFO\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3125
 [CREATE\_GO\_SVNVERSION\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3121
 [CREATE\_INIT\_PY]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/create_init_py.py?rev=5002020#L6
 [CREATE\_INIT\_PY\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/create_init_py.py?rev=5002020#L40
 [CREATE\_INIT\_PY\_STRUCTURE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/create_init_py.py?rev=5002020#L29
 [CREATE\_JAVA\_SVNVERSION\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3117
 [CREATE\_SVNVERSION\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3113
 [CTEMPLATE\_VARNAMES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3684
 [CUDA\_NVCC\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3231
 [CXXFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3225
 [DARWIN\_SIGNED\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4329
 [DARWIN\_STRINGS\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4325
 [DATA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1039
 [DEB\_VERSION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3376
 [DECIMAL\_MD5\_LOWER\_32\_BITS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3137
 [DECLARE\_EXTERNAL\_HOST\_RESOURCES\_BUNDLE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [DECLARE\_EXTERNAL\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [DEFAULT]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [DEPENDENCY\_MANAGEMENT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2237
 [DEPENDS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1070
 [DISABLE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [DLL\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [DOCS\_CONFIG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1881
 [DOCS\_DIR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1872
 [DOCS\_VARS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1890
 [DUMPERF\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3185
 [ELSE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [ELSEIF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [ENABLE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [END]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [ENDIF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [ENV]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1078
 [EXCLUDE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2164
 [EXCLUDE\_TAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [EXPORTS\_SCRIPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L914
 [EXPORT\_MAPKIT\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3830
 [EXPORT\_YMAPS\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3667
 [EXTERNAL\_JAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2210
 [EXTERNAL\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [EXTRADIR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [EXTRALIBS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [EXTRALIBS\_STATIC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2526
 [FAT\_RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/res.py?rev=5002020#L31
 [FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/files.py?rev=5002020#L1
 [FLAT\_JOIN\_SRCS\_GLOBAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2677
 [FORK\_SUBTESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2632
 [FORK\_TESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2620
 [FORK\_TEST\_FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2654
 [FROM\_MDS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3639
 [FROM\_SANDBOX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3630
 [FUZZ\_DICTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1003
 [FUZZ\_OPTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1020
 [GENERATED\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3717
 [GENERATE\_ENUM\_SERIALIZATION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3363
 [GENERATE\_ENUM\_SERIALIZATION\_WITH\_HEADER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3368
 [GENERATE\_PY\_EVS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2578
 [GENERATE\_PY\_PROTOS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2566
 [GENERATE\_SCRIPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/java.py?rev=5002020#L50
 [GEN\_SCHEEME2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3507
 [GLOBAL\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1662
 [GO\_ASM\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3991
 [GO\_CGO1\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3998
 [GO\_CGO2\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4005
 [GO\_COMPILE\_CGO1]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4041
 [GO\_COMPILE\_CGO2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4045
 [GO\_COMPILE\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4012
 [GO\_COMPILE\_SYMABIS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4037
 [GO\_FAKE\_OUTPUT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4067
 [GO\_GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4130
 [GO\_LDFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4111
 [GO\_LINK\_EXE\_IMPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4053
 [GO\_LINK\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4019
 [GO\_LINK\_LIB\_IMPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4049
 [GO\_LINK\_TEST\_IMPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4057
 [GO\_PACKAGE\_NAME]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4078
 [GO\_PROTOC\_PLUGIN\_ARGS\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L358
 [GO\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L362
 [GO\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4083
 [GO\_TEST\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [GO\_TEST\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4089
 [GO\_UNUSED\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4100
 [GO\_XTEST\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4095
 [GO\_YDB\_GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4139
 [GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L462
 [IDEA\_EXCLUDE\_DIRS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2130
 [IDEA\_RESOURCE\_DIRS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2135
 [IF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [IMPORT\_YMAPS\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3659
 [INCLUDE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [INCLUDE\_TAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [INDUCED\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [IOS\_APP\_ASSETS\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4321
 [IOS\_APP\_COMMON\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4316
 [IOS\_APP\_SETTINGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ios_app_settings.py?rev=5002020#L5
 [IOS\_ASSETS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ios_assets.py?rev=5002020#L6
 [JAVAC\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2140
 [JAVA\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L432
 [JAVA\_IGNORE\_CLASSPATH\_CLASH\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4251
 [JAVA\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/java.py?rev=5002020#L67
 [JAVA\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L401
 [JAVA\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2196
 [JAVA\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L595
 [JAVA\_TEST\_DEPS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L655
 [JOINSRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3326
 [JOIN\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2667
 [JOIN\_SRCS\_GLOBAL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2672
 [JVM\_ARGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2069
 [LDFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3192
 [LDFLAGS\_FIXED]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3197
 [LICENSE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3454
 [LINT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1113
 [LJ\_21\_ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/lj_archive.py?rev=5002020#L19
 [LJ\_ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/lj_archive.py?rev=5002020#L1
 [LLVM\_BC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/llvm_bc.py?rev=5002020#L6
 [LLVM\_COMPILE\_C]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3727
 [LLVM\_COMPILE\_CXX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3722
 [LLVM\_COMPILE\_LL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3732
 [LLVM\_LINK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3737
 [LLVM\_OPT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3742
 [LSAN\_SUPPRESSIONS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/lsan_suppressions.py?rev=5002020#L1
 [LUA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3565
 [LUAJIT\_21\_OBJDUMP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3336
 [LUAJIT\_OBJDUMP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3331
 [MACROS\_WITH\_ERROR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/macros_with_error.py?rev=5002020#L4
 [MAPKITIDL]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/plugins/plugin_mapkitidl_handler.cpp?rev=5002020#L396
 [MAPKIT\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3808
 [MAPKIT\_CPP\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L383
 [MAPKIT\_ENABLE\_WHOLE\_ARCHIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3845
 [MAPSMOBI\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3823
 [MASMFLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3212
 [MAVEN\_GROUP\_ID]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2147
 [MESSAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [METAQUERYFILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/mq.py?rev=5002020#L12
 [MSVC\_FLAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4493
 [MX\_BIN\_TO\_INFO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3340
 [MX\_FORMULAS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/mx_archive.py?rev=5002020#L1
 [MX\_GEN\_TABLE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3352
 [NEED\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3431
 [NEED\_REVIEW]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3444
 [NO\_BUILD\_IF]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [NO\_CHECK\_IMPORTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3775
 [NO\_CLANG\_COVERAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3322
 [NO\_CODENAVIGATION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3273
 [NO\_COMPILER\_WARNINGS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3243
 [NO\_DEBUG\_INFO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3655
 [NO\_GPL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3461
 [NO\_JOIN\_SRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3307
 [NO\_LIBC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3300
 [NO\_LINT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1106
 [NO\_NEED\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3437
 [NO\_OPTIMIZE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3239
 [NO\_OPTIMIZE\_PY\_PROTOS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L327
 [NO\_PLATFORM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3293
 [NO\_PLATFORM\_RESOURCES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3269
 [NO\_PYTHON\_INCLUDES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2337
 [NO\_RUNTIME]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3286
 [NO\_SANITIZE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3314
 [NO\_SANITIZE\_COVERAGE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3318
 [NO\_SSE4]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2742
 [NO\_UTIL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3277
 [NO\_WERROR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3261
 [NO\_WSHADOW]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3265
 [ONLY\_TAGS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [OPTIMIZE\_PY\_PROTOS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L323
 [OWNER]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [PACK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1771
 [PACKAGE\_STRICT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1776
 [PARTITIONED\_RECURSE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [PARTITIONED\_RECURSE\_FOR\_TESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [PARTITIONED\_RECURSE\_ROOT\_RELATIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [PEERDIR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [PIRE\_INLINE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3082
 [PIRE\_INLINE\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3078
 [PRINT\_MODULE\_TYPE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/print_module_type.py?rev=5002020#L1
 [PROCESS\_DOCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/docs.py?rev=5002020#L30
 [PROCESS\_DOCSLIB]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/docs.py?rev=5002020#L26
 [PROTO2FBS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L411
 [PROTO\_PLUGIN\_ARGS\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L331
 [PROVIDES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2685
 [PY3\_COMPILE\_BYTECODE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3064
 [PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3591
 [PYTHON2\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1483
 [PYTHON3\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2387
 [PYTHON3\_MODULE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1494
 [PYTHON\_ADDINCL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2349
 [PYTHON\_PATH]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1093
 [PY\_CODENAV]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3779
 [PY\_COMPILE\_BYTECODE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3060
 [PY\_DOCTESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/pybuild.py?rev=5002020#L435
 [PY\_EVLOG\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L424
 [PY\_EVLOG\_CMD\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L419
 [PY\_MAIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/pybuild.py?rev=5002020#L491
 [PY\_NAMESPACE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1900
 [PY\_PROTOS\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [PY\_PROTO\_CMD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L393
 [PY\_PROTO\_CMD\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L389
 [PY\_REGISTER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/pybuild.py?rev=5002020#L453
 [PY\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4255
 [RECURSE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [RECURSE\_FOR\_TESTS]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [RECURSE\_ROOT\_RELATIVE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [REGISTER\_YQL\_PYTHON\_UDF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yql_python_udf.py?rev=5002020#L10
 [REQUIREMENTS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1060
 [RESOURCE]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/plugins/plugin_resource_handler.cpp?rev=5002020#L42
 [RESOURCE\_FILES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/res.py?rev=5002020#L47
 [RUN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L775
 [RUN\_ANTLR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/antlr.py?rev=5002020#L4
 [RUN\_ANTLR4]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/antlr4.py?rev=5002020#L6
 [RUN\_ANTLR4\_CPP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/antlr4.py?rev=5002020#L18
 [RUN\_JAVA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/run_java.py?rev=5002020#L5
 [RUN\_JAVA\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/java.py?rev=5002020#L33
 [RUN\_PROGRAM]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3539
 [RUN\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3802
 [SET]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [SETUP\_EXECTEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L781
 [SETUP\_PYTEST\_BIN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L765
 [SETUP\_RUN\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L789
 [SET\_APPEND]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [SET\_APPEND\_WITH\_GLOBAL]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [SIZE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2663
 [SKIP\_TEST]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1098
 [SOURCE\_GROUP]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [SPLIT\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/split_codegen.py?rev=5002020#L9
 [SPLIT\_CODEGEN\_BASE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3166
 [SPLIT\_DWARF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2506
 [SPLIT\_FACTOR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2643
 [SRC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2921
 [SRCDIR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2944
 [SRC\_CPP\_AVX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2990
 [SRC\_CPP\_AVX2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2994
 [SRC\_CPP\_SSE2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2974
 [SRC\_CPP\_SSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2978
 [SRC\_CPP\_SSE41]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2986
 [SRC\_CPP\_SSSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2982
 [SRC\_C\_AVX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2966
 [SRC\_C\_AVX2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2970
 [SRC\_C\_SSE2]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2950
 [SRC\_C\_SSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2954
 [SRC\_C\_SSE41]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2962
 [SRC\_C\_SSSE3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2958
 [STRIP]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3235
 [STRUCT\_CODEGEN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3180
 [SYMLINK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3513
 [SYSTEM\_PROPERTIES]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2064
 [TAG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1050
 [TASKLET]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3851
 [TASKLET\_REG]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3870
 [TEST\_CWD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2090
 [TEST\_DATA]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/ytest.py?rev=5002020#L34
 [TEST\_SRCS]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1025
 [TGZ\_FOR]: https://a.yandex-team.ru/arc/trunk/arcadia/devtools/ymake/yndex/builtin.cpp?rev=5002020#L18
 [TIMEOUT]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2606
 [TOUCH]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3758
 [UBERJAR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2103
 [UBERJAR\_HIDE\_EXCLUDE\_PATTERN]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2113
 [UBERJAR\_HIDING\_PREFIX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2108
 [UBERJAR\_PATH\_EXCLUDE\_PREFIX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2118
 [USE\_ERROR\_PRONE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2079
 [USE\_LINKER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4520
 [USE\_LINKER\_BFD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4536
 [USE\_LINKER\_GOLD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4539
 [USE\_LINKER\_LLD]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4542
 [USE\_PERL\_LIB]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2446
 [USE\_PYTHON]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2422
 [USE\_PYTHON3]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2432
 [USE\_RECIPE]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1087
 [USE\_YQL]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1642
 [VERSION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3465
 [WERROR]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3252
 [WITH\_JDK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L2242
 [XS\_PROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L405
 [YABS\_GENERATE\_CONF]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_generate_conf.py?rev=5002020#L10
 [YABS\_GENERATE\_PHANTOM\_CONF\_PATCH]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_generate_conf.py?rev=5002020#L35
 [YABS\_GENERATE\_PHANTOM\_CONF\_TEST\_CHECK]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_generate_conf.py?rev=5002020#L53
 [YABS\_SERVER\_BUILD\_YSON\_INDEX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_server_build_index.py?rev=5002020#L121
 [YABS\_SERVER\_PREPARE\_QXL\_FROM\_SANDBOX]: https://a.yandex-team.ru/arc/trunk/arcadia/build/plugins/yabs_server_build_index.py?rev=5002020#L92
 [YDB\_GRPC]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L4135
 [YMAPS\_GENERATE\_SPROTO\_HEADER]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3672
 [YMAPS\_SPROTO]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L3677
 [YQL\_ABI\_VERSION]: https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=5002020#L1708
