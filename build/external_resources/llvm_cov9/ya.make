# Remove this library when clang9 will be default toolchain in the Arcadia
RESOURCES_LIBRARY()



IF (HOST_OS_LINUX)
    DECLARE_EXTERNAL_RESOURCE(LLVM_COV9 sbr:1837932629)
ENDIF()

END()
