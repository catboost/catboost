# CatBoost documentation

Welcome to the CatBoost [docs](https://catboost.ai/docs/).

## About the docs

The documentation is developed using [Yandex Flavored Markdown](https://github.com/yandex-cloud/yfm-docs) (YFM). See [YFM syntax](https://github.com/yandex-cloud/yfm-transform/blob/master/DOCS.md).

## Making the CatBoost docs better

To contribute to the CatBoost docs you need to first read the [Yandex Contributor License Agreement](https://github.com/catboost/catboost/blob/master/CONTRIBUTING.md) (CLA) text and add to your pull request, that you agree to the terms of the CLA. Instructions for contributors can be found [here](https://catboost.ai/docs/concepts/development-and-contributions.html).

If you notice a typo or error in the documentation or want to add any section, create a pull request (PR) with edits via GitHub.

## Building the CatBoost docs

Before creating a pull request, build the docs locally for checking your changes. To do this, use the [yfm-docs](https://github.com/yandex-cloud/yfm-docs) tool.

1. Install **yfm-docs**:

   `npm i @doc-tools/docs -g`

   To update the version of **yfm-docs**, use the  `npm i @doc-tools/docs@latest -g` command.

1. Build the docs:

   `yfm -i ./catboost/docs -o ./docs-gen --varsPreset "external" -c -en --lang=en`, where `docs` is a folder with the source texts, `docs-gen` is a folder with the generated documentation.

## Licenses

© YANDEX LLC, 2017-2024. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
