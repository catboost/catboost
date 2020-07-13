

RECURSE(
    antlr3
    antlr4
    gradle
    maven
)

IF (OS_ANDROID)
    RECURSE(
    android_sdk
    mapsmobi_maven_repo
)
ENDIF()

IF (OS_IOS)
    RECURSE(
    mapsmobi_ios_pods
)
ENDIF()

IF (OS_LINUX)
    RECURSE(
    codenavigation
    llvm_cov9
)
ENDIF()

IF (OS_DARWIN OR OS_LINUX OR OS_WINDOWS)
    RECURSE(
    arcadia_cpp_styleguide
    arcadia_grpc_cpp
    arcadia_grpc_java
    arcadia_grpc_python
    arcadia_protoc
    arcadia_protoc_std
    arcadia_protoc-gen-go
    arcadia_python3_pycc
    arcadia_ragel6
    arcadia_rescompiler
    arcadia_rescompressor
    arcadia_rorescompiler
    arcadia_yasm
    flake8_py2
    flake8_py3
    flakes_py2
    flakes_py3
    go_tools
    goyndexer
    pep8_py2
    pep8_py3
    protoc-gen-javalite
    ymake
    yolint
)
ENDIF()
