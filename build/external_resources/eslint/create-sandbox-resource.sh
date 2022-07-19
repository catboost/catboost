#!/bin/bash
set -e

ESLINT_VERSION="7.27.0"
PACKAGES="eslint@$ESLINT_VERSION @yandex-int/eslint-plugin-config@1.1.0"

echo "Creating temporary directory…"
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
echo "Temporary directory ${TMP_DIR} is created"

echo "Installing packages…"
ESLINT_DIR="eslint-resource"
mkdir $ESLINT_DIR && cd $ESLINT_DIR
npm init -y
npm install --save-dev --save-exact --registry=https://npm.yandex-team.ru ${PACKAGES}
echo "Packages are installed"

echo "Creating archive…"
RESOURCE_DIR="node_modules"
ARCHIVE="eslint-$ESLINT_VERSION.tar.gz"
tar --create --gzip --file=$ARCHIVE $RESOURCE_DIR
echo "Archive $(pwd)/$ARCHIVE is created"

echo "Uploading file ${ARCHIVE}…"
DESCRIPTION="Bundle for https://st.yandex-team.ru/FBP-84. Content: ${PACKAGES}."
ya upload $ARCHIVE -d="${DESCRIPTION}" --ttl="inf" --attr="eslint=${ESLINT_VERSION}"
echo "File $ARCHIVE is uploaded"

echo "Cleaning up…"
rm -rf $TMP_DIR

echo "Done"
