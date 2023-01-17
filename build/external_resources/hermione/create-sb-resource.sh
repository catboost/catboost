#!/bin/bash
set -e

HERMIONE_VERSION="5.0.0"
HERMIONE_CLI_VERSION="1.2.4"

# Plugins
HTML_REPORTER_VERSION="9.0.0"
HERMIONE_CHUNKS_VERSION="0.2.1"
HERMIONE_PASSIVE_BROWSERS_VERSION="0.1.1"

PACKAGES="hermione@$HERMIONE_VERSION
@yandex-int/hermione-cli@$HERMIONE_CLI_VERSION
html-reporter@$HTML_REPORTER_VERSION
hermione-chunks@$HERMIONE_CHUNKS_VERSION
hermione-passive-browsers@$HERMIONE_PASSIVE_BROWSERS_VERSION"

echo "Creating temporary directory…"
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
echo "Temporary directory ${TMP_DIR} is created"

echo "Installing packages…"
HERMIONE_DIR="hermione-resource"
mkdir $HERMIONE_DIR && cd $HERMIONE_DIR
npm init -y
npm install --save-dev --save-exact --registry=https://npm.yandex-team.ru ${PACKAGES}
echo "Packages are installed"

echo "Creating archive…"
RESOURCE_DIR="node_modules"
ARCHIVE="hermione-$HERMIONE_VERSION.tar.gz"
tar --create --gzip --file=$ARCHIVE $RESOURCE_DIR
echo "Archive $(pwd)/$ARCHIVE is created"

echo "Uploading file ${ARCHIVE}…"
DESCRIPTION="Bundle for run hermione in tier0. Content: ${PACKAGES}"
ya upload $ARCHIVE -d="${DESCRIPTION}" --ttl="inf" --attr="hermione=${HERMIONE_VERSION}" --attr="hermione-cli=${HERMIONE_CLI_VERSION}" --attr="html-reporter=${HTML_REPORTER_VERSION}" --attr="hermione-chunks=${HERMIONE_CHUNKS_VERSION}" --attr="hermione-passive-browsers=${HERMIONE_PASSIVE_BROWSERS_VERSION}"
echo "File $ARCHIVE is uploaded"

echo "Cleaning up…"
rm -rf $TMP_DIR

echo "Done"
