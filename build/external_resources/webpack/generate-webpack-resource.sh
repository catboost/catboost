#!/bin/bash
set -e

WEBPACK_VERSION="5.69.1"
PACKAGES="webpack@$WEBPACK_VERSION webpack-cli@4.9.2 ts-loader@9.2.6 webpack-dev-middleware@5.3.3 typescript@4.5.5"

cd ~

TMP_DIR="tmp"
if [[ ! -d $TMP_DIR ]]
then
    mkdir $TMP_DIR
fi
cd $TMP_DIR

WEBPACK_DIR="webpack-resource"
mkdir $WEBPACK_DIR && cd $WEBPACK_DIR

npm init -y

npm install --save-dev --save-exact --registry=https://npm.yandex-team.ru ${PACKAGES}
echo "Packages installed successfully"

RESOURCE_DIR="node_modules"
ARCHIVE="webpack-$WEBPACK_VERSION.tar.gz"
tar --create --gzip --file=$ARCHIVE $RESOURCE_DIR
echo "Created file $(pwd)/$ARCHIVE"

DESCR="Bundle for https://st.yandex-team.ru/FEI-24499. Content: ${PACKAGES}"
ya upload $ARCHIVE -d="${DESCR}" --ttl="inf" --attr="webpack=${WEBPACK_VERSION}"
echo "$ARCHIVE uploaded successfully"

echo "Cleanup…"
rm -fr ~/tmp/$WEBPACK_DIR

echo "Done."
