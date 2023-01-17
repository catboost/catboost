#!/bin/bash

set -e -o pipefail

ARC_ROOT=$(arc root)
ya make $ARC_ROOT/tools/nots
$ARC_ROOT/tools/nots/nots create-external-resource --skip-external-resources-meta $@
