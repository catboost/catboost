# CMake generated Testfile for 
# Source directory: /Users/makar/Documents/Course work/Repository/catboost/catboost/libs/model/ut
# Build directory: /Users/makar/Documents/Course work/Repository/catboost/build_artifacts/catboost/libs/model/ut
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(model_ut "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/catboost/libs/model/ut/model_ut" "--print-before-suite" "--print-before-test" "--fork-tests" "--print-times" "--show-fails")
set_tests_properties(model_ut PROPERTIES  ENVIRONMENT "" LABELS "MEDIUM" PROCESSORS "1" _BACKTRACE_TRIPLES "/Users/makar/Documents/Course work/Repository/catboost/cmake/common.cmake;295;add_test;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/model/ut/CMakeLists.darwin-arm64.txt;65;add_yunittest;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/model/ut/CMakeLists.darwin-arm64.txt;0;;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/model/ut/CMakeLists.txt;30;include;/Users/makar/Documents/Course work/Repository/catboost/catboost/libs/model/ut/CMakeLists.txt;0;")
subdirs("lib")
