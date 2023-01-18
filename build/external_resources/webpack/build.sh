#!/bin/bash

set -e -o pipefail

ARC_ROOT=$(arc root)
NOTS_DIR=$ARC_ROOT/devtools/frontend_build_platform/nots
ya make $NOTS_DIR/er_importer
$NOTS_DIR/er_importer/er_importer --external-resources-meta $NOTS_DIR/cli/external-resources-meta.json $@

