EXTERNAL_JAVA_LIBRARY()



SRCS(
    src/main/java/ru/yandex/library/svnversion/VcsVersion.java
)

LINT(strict)

END()

RECURSE(
    tests
)
