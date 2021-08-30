

RECURSE(
    antlr3
    antlr4
    gradle
    maven
    swift-demangle
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
)
ENDIF()

IF (OS_DARWIN OR OS_LINUX OR OS_WINDOWS)
    RECURSE(
    flake8_py2
    flake8_py3
    go_fake_xcrun
    go_tools
    goyndexer
    pnpm
    typescript
    ymake
    yolint
)
ENDIF()
