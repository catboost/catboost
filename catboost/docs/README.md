# CatBoost documentation

Welcome to the CatBoost [docs](https://catboost.ai/docs/).

## About the docs

The documentation is developed using [Yandex Flavored Markdown](https://diplodoc.com/docs/en/index-yfm) (YFM). See [YFM syntax](https://diplodoc.com/docs/en/syntax/).

## Making the CatBoost docs better

If you notice a typo or error in the documentation or want to add any section, create a pull request (PR) with edits via GitHub.

## Building the CatBoost docs

Before creating a pull request, build the docs locally for checking your changes. To do this, use the [yfm-docs](https://github.com/diplodoc-platform/cli) tool.

1. Install **yfm-docs**:

   `npm i @diplodoc/cli -g`

   To update the version of **yfm-docs**, use the  `npm i @diplodoc/cli@latest -g` command.

1. Build the docs:

   `yfm -i ./catboost/docs -o ./docs-gen --varsPreset "external" -c -en --lang=en`, where `docs` is a folder with the source texts, `docs-gen` is a folder with the generated documentation.

## Licenses

© YANDEX LLC, 2017-2026. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
