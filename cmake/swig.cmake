set(SWIG_EXECUTABLE ${CMAKE_BINARY_DIR}/bin/swig${CMAKE_EXECUTABLE_SUFFIX})
set(SWIG_SOURCE_FILE_EXTENSIONS .swg)

function(add_swig_jni_library TgtName)
  set(opts "")
  set(oneval_args GEN_JAVA_FILES_LIST)
  set(multival_args SOURCES)
  cmake_parse_arguments(SWIG_JNI_LIB
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )

  set_property(SOURCE
    ${SWIG_JNI_LIB_SOURCES}
    PROPERTY
      CPLUSPLUS On
  )

  swig_add_library(${TgtName}
    TYPE SHARED
    LANGUAGE java
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java
    OUTFILE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpp
    SOURCES 
      ${SWIG_JNI_LIB_SOURCES}
  )
  add_custom_command(TARGET
    ${TgtName}
    POST_BUILD COMMAND
      ${CMAKE_COMMAND} -DJAVA_SRC_DIR=${CMAKE_CURRENT_BINARY_DIR}/java -DJAVA_LST=${CMAKE_CURRENT_BINARY_DIR}/swig_gen_java.lst -P ${CMAKE_SOURCE_DIR}/build/scripts/gather_swig_java.cmake
    BYPRODUCTS ${SWIG_JNI_LIB_GEN_JAVA_FILES_LIST}
  )
endfunction()
