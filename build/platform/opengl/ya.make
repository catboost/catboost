RESOURCES_LIBRARY()



IF (NOT OPENGL_REQUIRED)
    MESSAGE(FATAL_ERROR "No OpenGL Toolkit for your build")
ELSE()
    IF (OS_LINUX)
        IF (OPENGL_VERSION STREQUAL "18.0.5")
            DECLARE_EXTERNAL_RESOURCE(OPENGL sbr:1271121094)
            SET(OS_SUFFIX "x86_64-linux-gnu")
        ELSE()
            ENABLE(OPENGL_NOT_FOUND)
        ENDIF()
    ELSE()
        ENABLE(OPENGL_NOT_FOUND)
    ENDIF()

    IF (OPENGL_NOT_FOUND)
        MESSAGE(FATAL_ERROR "No OpenGL Toolkit for the selected platform")
    ELSE()
        CFLAGS(GLOBAL "-I$OPENGL_RESOURCE_GLOBAL/usr/include")
        LDFLAGS("-L$OPENGL_RESOURCE_GLOBAL/usr/lib/$OS_SUFFIX")
    ENDIF()
ENDIF()

END()
