<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Release Process

## Overview

This file documents the release process for the `object_store` crate.

We release a new version of `object_store` according to the schedule listed in
the [main README.md]

[main README.md]: https://github.com/apache/arrow-rs-object-store?tab=readme-ov-file#release-schedule

As we are still in an early phase, we use the 0.x version scheme. If any code has
been merged to main that has a breaking API change, as defined in [Rust RFC 1105]
the minor version number is incremented changed (e.g. `0.3.0` to `0.4.0`).
Otherwise the patch version is incremented (e.g. `0.3.0` to `0.3.1`).

[Rust RFC 1105]: https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md

# Release Mechanics

## Process Overview

As part of the Apache governance model, official releases consist of
signed source tarballs approved by the PMC.

We then use the code in the approved source tarball to release to
crates.io, the Rust ecosystem's package manager.

We create a `CHANGELOG.md` so our users know what has been changed between releases.

The CHANGELOG is created automatically using
[update_change_log.sh](https://github.com/apache/arrow-rs-object-store/blob/main/dev/release/update_change_log.sh)

This script creates a changelog using github issues and the
labels associated with them.

## Prepare CHANGELOG and version:

Now prepare a PR to update `CHANGELOG.md` and versions on `main` to reflect the planned release.

See [#437] for an example.

[#437]: https://github.com/apache/arrow-rs-object-store/pull/437

```bash
git checkout main
git pull
git checkout -b <RELEASE_BRANCH>

# Update versions. Make sure to run it before the next step since we do not want CHANGELOG-old.md affected.
sed -i '' -e 's/0.11.0/0.11.1/g' `find . -name 'Cargo.toml' -or -name '*.md' | grep -v CHANGELOG`
git commit -a -m 'Update version'

# ensure your github token is available
export CHANGELOG_GITHUB_TOKEN=<TOKEN>

# manually edit ./dev/release/update_change_log.sh to reflect the release version
# create the changelog
./dev/release/update_change_log.sh

# review change log / and edit associated issues and labels if needed, rerun update_change_log.sh

# Commit changes
git commit -a -m 'Create changelog'

# push changes to fork and create a PR to main
git push
```

Note that when reviewing the change log, rather than editing the
`CHANGELOG.md`, it is preferred to update the issues and their labels
(e.g. add `invalid` label to exclude them from release notes)

Merge this PR to `main` prior to the next step.

## Prepare release candidate tarball

After you have merged the updates to the `CHANGELOG` and version,
create a release candidate using the following steps. Note you need to
be a committer to run these scripts as they upload to the apache `svn`
distribution servers.

### Create git tag for the release:

While the official release artifact is a signed tarball, we also tag the commit it was created for convenience and code archaeology.

Use a string such as `v0.4.0` as the `<version>`.

Create and push the tag thusly:

```shell
git fetch apache
git tag <version> apache/main
# push tag to apache
git push apache <version>
```

### Pick an Release Candidate (RC) number

Pick numbers in sequential order, with `1` for `rc1`, `2` for `rc2`, etc.

### Create, sign, and upload tarball

Run `create-tarball.sh` with the `<version>` tag and `<rc>` and you found in previous steps.

```shell
./dev/release/create-tarball.sh 0.11.1 1
```

The `create-tarball.sh` script

1. creates and uploads a release candidate tarball to the [arrow
   dev](https://dist.apache.org/repos/dist/dev/arrow) location on the
   apache distribution svn server

2. provide you an email template to
   send to dev@arrow.apache.org for release voting.

### Vote on Release Candidate tarball

Send an email, based on the output from the script to dev@arrow.apache.org. The email should look like

```
Draft email for dev@arrow.apache.org mailing list

---------------------------------------------------------
To: dev@arrow.apache.org
Subject: [VOTE][RUST] Release Apache Arrow Rust Object Store 0.11.1 RC1

Hi,

I would like to propose a release of Apache Arrow Rust Object
Store Implementation, version 0.11.1.

This release candidate is based on commit: b945b15de9085f5961a478d4f35b0c5c3427e248 [1]

The proposed release tarball and signatures are hosted at [2].

The changelog is located at [3].

Please download, verify checksums and signatures, run the unit tests,
and vote on the release. There is a script [4] that automates some of
the verification.

The vote will be open for at least 72 hours.

[ ] +1 Release this as Apache Arrow Rust Object Store
[ ] +0
[ ] -1 Do not release this as Apache Arrow Rust Object Store  because...

[1]: https://github.com/apache/arrow-rs-object-store/tree/b945b15de9085f5961a478d4f35b0c5c3427e248
[2]: https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-object-store-rs-0.11.1-rc1/
[3]: https://github.com/apache/arrow-rs-object-store/blob/b945b15de9085f5961a478d4f35b0c5c3427e248/CHANGELOG.md
[4]: https://github.com/apache/arrow-rs-object-store/blob/main/dev/release/verify-release-candidate.sh
```

For the release to become "official" it needs at least three Apache Arrow PMC members to vote +1 on it.

## Verifying release candidates

The `dev/release/verify-release-candidate.sh` script can assist in the verification process. Run it like:

```
./dev/release/verify-release-candidate.sh 0.11.0 1
```

#### If the release is not approved

If the release is not approved, fix whatever the problem is and try again with the next RC number

### If the release is approved,

Move tarball to the release location in SVN, e.g. https://dist.apache.org/repos/dist/release/arrow/apache-arrow-object-store-rs-4.1.0-rc4/, using the `release-tarball.sh` script:

```shell
./dev/release/release-tarball.sh 4.1.0 2
```

Congratulations! The release is now official!

### Publish on Crates.io

Only approved releases of the tarball should be published to
crates.io, in order to conform to Apache Software Foundation
governance standards.

An Arrow committer can publish this crate after an official project release has
been made to crates.io using the following instructions.

Follow [these
instructions](https://doc.rust-lang.org/cargo/reference/publishing.html) to
create an account and login to crates.io before asking to be added as an owner
of the [object store crate](https://crates.io/crates/object_store).

Download and unpack the official release tarball

Verify that the Cargo.toml in the tarball contains the correct version
(e.g. `version = "0.11.0"`) and then publish the crate with the
following commands

```shell
cargo publish
```
