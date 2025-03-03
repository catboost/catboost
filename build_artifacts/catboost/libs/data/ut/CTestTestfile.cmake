# CMake generated Testfile for 
# Source directory: /Users/makar/Documents/Course work/Repository/catboost/catboost/libs/data/ut
# Build directory: /Users/makar/Documents/Course work/Repository/catboost/build_artifacts/catboost/libs/data/ut
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(catboost-libs-data-ut "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/catboost/libs/data/ut/catboost-libs-data-ut" "--print-before-suite" "--print-before-test" "--fork-tests" "--print-times" "--show-fails")
set_tests_properties(catboost-libs-data-ut PROPERTIES  ENVIRONMENT "" LABELS "SMALL" PROCESSORS "1" _BACKTRACE_TRIPLES "/Users/makar/Documents/Course work/Repository/catboost/cmake/common.cmake;295;add_test;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/data/ut/CMakeLists.darwin-arm64.txt;80;add_yunittest;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/data/ut/CMakeLists.darwin-arm64.txt;0;;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/data/ut/CMakeLists.txt;30;include;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/data/ut/CMakeLists.txt;0;")
subdirs("lib")
