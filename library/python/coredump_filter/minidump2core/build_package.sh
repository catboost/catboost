#!/usr/bin/env bash
rm -rf dist
python setup.py sdist upload -r yandex
