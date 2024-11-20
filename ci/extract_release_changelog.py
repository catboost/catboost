#!/usr/bin/env python

import argparse
import re


def extract_release_changelog(all_changelog: str, dst_release_changelog: str, release_version: str):
    in_release_changelog = False
    with open(all_changelog) as src:
        with open(dst_release_changelog, 'w') as dst:
            for l in src:
                if in_release_changelog:
                    if re.match('^# Release.*', l[:-1]):
                        return
                    dst.write(l)
                elif re.match(f'^# Release {release_version}$', l[:-1]):
                    in_release_changelog = True

    if not in_release_changelog:
        raise Exception(f'Release {release_version} has not been found in changelog')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-changelog', help='file with all changelog', required=True)
    parser.add_argument('--dst-release-changelog', help='file with extracted release changelog', required=True)
    parser.add_argument('--release-version', help='release version', required=True)
    parsed_args = parser.parse_args()

    extract_release_changelog(**vars(parsed_args))
