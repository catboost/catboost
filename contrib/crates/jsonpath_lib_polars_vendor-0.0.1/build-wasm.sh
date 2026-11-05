#!/bin/bash

set -e

# project_root
DIR="$(pwd)"
WASM="${DIR}"/wasm
WASM_WWW="${WASM}"/www
WASM_WWW_BENCH="${WASM}"/www_bench
WASM_NODEJS_PKG="${WASM}"/nodejs_pkg
WASM_WEB_PKG="${WASM}"/web_pkg
WASM_TEST="${WASM}"/tests
DOCS="${DIR}"/docs
DOCS_BENCH="${DOCS}"/bench

__msg () {
    echo ">>>>>>>>>>$1<<<<<<<<<<"
}

__cargo_clean () {
    rm -f "${DIR}"/Cargo.lock
    rm -f "${WASM}"/Cargo.lock
    cd "${WASM}" && cargo clean && \
    cd "${DIR}" && cargo clean
}

echo
__msg "clean wasm"
rm -rf \
    "${WASM_NODEJS_PKG}" \
    "${WASM_WEB_PKG}" \
    "${WASM_WWW}"/dist \
    "${WASM_WWW}"/node_modules \
    "${WASM_WWW}"/package-lock.json \
    "${WASM_WWW_BENCH}"/dist \
    "${WASM_WWW_BENCH}"/node_modules \
    "${WASM_WWW_BENCH}"/package-lock.json \
    "${WASM_TEST}"/node_modules \
    "${WASM_TEST}"/package-lock.json

__msg "clean cargo clean"
__cargo_clean

echo
wasm_pack_version=$(wasm-pack -V)
__msg "wasm-pack: ${wasm_pack_version}"

echo
__msg "wasm-pack nodejs"
cd "${WASM}" && wasm-pack build --release --target "nodejs" --out-dir "${WASM_NODEJS_PKG}"

__msg "npm install: wasm test"
cd "${WASM_TEST}" && npm install "${WASM_NODEJS_PKG}" && npm install

echo
__msg "wasm test"
cd "${WASM_TEST}" && npm test

if [ "$1" = "docs" ]; then
  echo
  __msg "wasm-pack web"
  cd "${WASM}" && wasm-pack build --release --out-dir "${WASM_WEB_PKG}"

  echo
  __msg "jsonpath-wasm npm link"
  cd "${WASM_WEB_PKG}" && npm link

  __msg "npm install: wasm"
  cd "${WASM_WWW}" && npm install
  __msg "npm install: wasm_bench"
  cd "${WASM_WWW_BENCH}" && npm install

  echo
  __msg "link"
  cd "${WASM_WWW}" && npm link jsonpath-wasm
  cd "${WASM_WWW_BENCH}" && npm link jsonpath-wasm

  echo
  __msg "docs"
  cd "${WASM_WWW}" && \
      npm run build &&
      rm -f "${DOCS}"/*.js "${DOCS}"/*.wasm "${DOCS}"/*.html && \
      cp "${WASM_WWW}"/dist/*.* "${DOCS}"/

  cd "${WASM_WWW_BENCH}" && \
      npm run build &&
      rm -f "${DOCS_BENCH}"/*.js "${DOCS_BENCH}"/*.wasm "${DOCS_BENCH}"/*.html && \
      cp "${WASM_WWW_BENCH}"/dist/*.* "${DOCS_BENCH}"/
fi

__msg "wasm done"