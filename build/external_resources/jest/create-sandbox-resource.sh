#!/bin/bash
set -e

JEST_VERSION="27.1.0"
PACKAGES="jest@$JEST_VERSION"

echo "Creating temporary directory…"
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
echo "Temporary directory ${TMP_DIR} is created"

echo "Installing packages…"
JEST_DIR="jest-resource"
mkdir $JEST_DIR && cd $JEST_DIR
npm init -y
npm install --save-dev --save-exact --registry=https://npm.yandex-team.ru ${PACKAGES}
echo "Packages are installed"

echo "Creating archive…"
RESOURCE_DIR="node_modules"
ARCHIVE="jest-$JEST_VERSION.tar.gz"
tar --create --gzip --file=$ARCHIVE $RESOURCE_DIR
echo "Archive $(pwd)/$ARCHIVE is created"

echo "Uploading file ${ARCHIVE}…"
DESCRIPTION="Bundle for https://st.yandex-team.ru/FBP-59. Content: ${PACKAGES}."
ya upload $ARCHIVE -d="${DESCRIPTION}" --ttl="inf" --attr="jest=${JEST_VERSION}"
echo "File $ARCHIVE is uploaded"

echo "Cleaning up…"
rm -rf $TMP_DIR

echo "Done"
