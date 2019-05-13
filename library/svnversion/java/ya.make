EXTERNAL_JAVA_LIBRARY()



CREATE_JAVA_SVNVERSION_FOR(SvnConstants.java)

SRCS(
    src/main/java/ru/yandex/library/svnversion/SvnVersion.java
    src/main/java/ru/yandex/library/svnversion/VcsVersion.java
)

LINT(strict)

END()

RECURSE(
    tests
)
