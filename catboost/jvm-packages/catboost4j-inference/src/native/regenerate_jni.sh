#!/bin/bash
set -e
set -x

ARCADIA_ROOT="$(realpath ../../../..)"
YA_PATH="${ARCADIA_ROOT}/ya"
JAVA_PATH="$(dirname $(dirname $(${YA_PATH} tool java --print-path)))"
JAVAC_PATH="${JAVA_PATH}/bin/javac"
JAVAH_PATH="${JAVA_PATH}/bin/javah"

(>&2 echo "javac path: ${JAVAC_PATH}")

# So it turns out that as of Java 8+ `javah` is deprecated and you can just run `javac` with `-h`
# argument. But it doesn't work with .jar files?

cd ../src
${YA_PATH} make --quiet
${JAVAH_PATH}     \
    -verbose      \
    -d ../jni     \
    -cp j-src.jar \
    -jni          \
    ai.catboost.CatBoostJNI
../jni/fix_jni_headers.py ../jni/ai_catboost_CatBoostJNI.h
(>&2 echo "Done")
