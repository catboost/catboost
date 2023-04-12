if(JNITarget_FIND_QUIETLY)
  find_package(JNI QUIET)
elseif(JNITarget_FIND_REQUIRED)
  find_package(JNI REQUIRED)
else()
  find_package(JNI)
endif()

set(JNI_TARGET_INCLUDE_DIRS ${JNI_INCLUDE_DIRS})
set(JNI_TARGET_LIBRARIES ${JNI_LIBRARIES})

if (JNI_FOUND)
  add_library(JNITarget::jni IMPORTED UNKNOWN)
  set_property(TARGET JNITarget::jni PROPERTY
    IMPORTED_LOCATION ${JAVA_JVM_LIBRARY}
  )
  set_property(TARGET JNITarget::jni PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2}
  )

  add_library(JNITarget::jni_awt IMPORTED UNKNOWN)
  set_property(TARGET JNITarget::jni_awt PROPERTY
    IMPORTED_LOCATION ${JAVA_AWT_LIBRARY}
  )
  set_property(TARGET JNITarget::jni_awt PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${JAVA_AWT_INCLUDE_PATH}
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JNITarget DEFAULT_MSG JNI_TARGET_LIBRARIES JNI_TARGET_INCLUDE_DIRS)

mark_as_advanced(JNI_TARGET_INCLUDE_DIRS JNI_TARGET_LIBRARIES)
