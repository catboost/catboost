#!/bin/bash

# Usage: makechangelog <git commit range>
#
# This script depends on and uses the github `gh` binary
# which needs to be authenticated to use.
#
# Will produce some duplicates for PRs integrated using rebase,
# but those will not occur with current merge queue.

git log --first-parent --pretty="tformat:%H" "$@" | while IFS= read -r commit_sha
do
    gh api "/repos/:owner/:repo/commits/${commit_sha}/pulls" \
        -q ".[] | \"- \(.title) by [@\(.user.login)](\(.user.html_url)) [#\(.number)](\(.html_url))\""
done | uniq

