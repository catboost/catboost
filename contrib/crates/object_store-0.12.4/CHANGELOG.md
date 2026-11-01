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

# Changelog

## [v0.12.4](https://github.com/apache/arrow-rs-object-store/tree/v0.12.4) (2025-09-19)

[Full Changelog](https://github.com/apache/arrow-rs-object-store/compare/v0.12.3...v0.12.4)

**Implemented enhancements:**

- Allow flagging `PUT` operations as idempotent. [\#464](https://github.com/apache/arrow-rs-object-store/issues/464)
- Release object store  `0.12.3` \(non breaking API\) Release July 2025 [\#428](https://github.com/apache/arrow-rs-object-store/issues/428)
- LocalFileSystem: offset for `list_with_offset` can't be identified  / List results \*must\* be sorted [\#388](https://github.com/apache/arrow-rs-object-store/issues/388)
- Support setting storage class when objects are written [\#330](https://github.com/apache/arrow-rs-object-store/issues/330)
- Support auth using AssumeRoleWithWebIdentity for non-AWS S3-compatible implementations [\#283](https://github.com/apache/arrow-rs-object-store/issues/283)
- Types from http through request leak into object\_store public interfaces but aren't re-exported [\#263](https://github.com/apache/arrow-rs-object-store/issues/263)

**Fixed bugs:**

- Retry does not cover connection errors [\#368](https://github.com/apache/arrow-rs-object-store/issues/368)

**Documentation updates:**

- Improve documentation for http client timeout [\#390](https://github.com/apache/arrow-rs-object-store/pull/390) ([alamb](https://github.com/alamb))

**Closed issues:**

- When a client http request is retried, I would like more information in the `info!` about the retry [\#486](https://github.com/apache/arrow-rs-object-store/issues/486)
- Range header causing AWS Signature issues [\#471](https://github.com/apache/arrow-rs-object-store/issues/471)
- Impossible to downcast an Error::Generic into a RetryError [\#469](https://github.com/apache/arrow-rs-object-store/issues/469)
- JWT session tokens cause SignatureDoesNotMatch with Supabase S3 [\#466](https://github.com/apache/arrow-rs-object-store/issues/466)
- Double url-encoding of special characters in key names [\#457](https://github.com/apache/arrow-rs-object-store/issues/457)
- Make `MultipartUpload` Sync [\#439](https://github.com/apache/arrow-rs-object-store/issues/439)
- Integrate HDFS object store [\#424](https://github.com/apache/arrow-rs-object-store/issues/424)
- Error performing POST when trying to write to S3 with a custom endpoint URL [\#408](https://github.com/apache/arrow-rs-object-store/issues/408)

**Merged pull requests:**

- Revert "refactor: remove AWS dynamo integration \(\#407\)" [\#493](https://github.com/apache/arrow-rs-object-store/pull/493) ([alamb](https://github.com/alamb))
- Fix for clippy 1.90 [\#492](https://github.com/apache/arrow-rs-object-store/pull/492) ([alamb](https://github.com/alamb))
- Add version 0.12.4 release plan to README [\#490](https://github.com/apache/arrow-rs-object-store/pull/490) ([alamb](https://github.com/alamb))
- chore\(client/retry\): include error info in logs when retry occurs [\#487](https://github.com/apache/arrow-rs-object-store/pull/487) ([philjb](https://github.com/philjb))
- AWS S3: Support STS endpoint, WebIdentity, RoleArn, RoleSession configuration [\#480](https://github.com/apache/arrow-rs-object-store/pull/480) ([Friede80](https://github.com/Friede80))
- build\(deps\): bump actions/github-script from 7 to 8 [\#478](https://github.com/apache/arrow-rs-object-store/pull/478) ([dependabot[bot]](https://github.com/apps/dependabot))
- build\(deps\): bump actions/setup-node from 4 to 5 [\#477](https://github.com/apache/arrow-rs-object-store/pull/477) ([dependabot[bot]](https://github.com/apps/dependabot))
- build\(deps\): bump actions/setup-python from 5 to 6 [\#476](https://github.com/apache/arrow-rs-object-store/pull/476) ([dependabot[bot]](https://github.com/apps/dependabot))
- chore: fix some clippy 1.89 warnings and ignore some doctests on wasm32 [\#468](https://github.com/apache/arrow-rs-object-store/pull/468) ([mbrobbel](https://github.com/mbrobbel))
- Allow "application\_credentials" in `impl FromStr for GoogleConfigKey` [\#467](https://github.com/apache/arrow-rs-object-store/pull/467) ([kylebarron](https://github.com/kylebarron))
- build\(deps\): bump actions/checkout from 4 to 5 [\#463](https://github.com/apache/arrow-rs-object-store/pull/463) ([dependabot[bot]](https://github.com/apps/dependabot))
- Add storage class for aws, gcp, and azure [\#456](https://github.com/apache/arrow-rs-object-store/pull/456) ([matthewmturner](https://github.com/matthewmturner))
- Remove use of deprecated StepRng from tests [\#449](https://github.com/apache/arrow-rs-object-store/pull/449) ([tustvold](https://github.com/tustvold))
- Fix not retrying connection errors [\#445](https://github.com/apache/arrow-rs-object-store/pull/445) ([johnnyg](https://github.com/johnnyg))
- Dont unwrap on body send [\#442](https://github.com/apache/arrow-rs-object-store/pull/442) ([cetra3](https://github.com/cetra3))
- feat: re-export HTTP types used in public API [\#441](https://github.com/apache/arrow-rs-object-store/pull/441) ([ByteBaker](https://github.com/ByteBaker))
- fix: update links in release docs and script [\#440](https://github.com/apache/arrow-rs-object-store/pull/440) ([mbrobbel](https://github.com/mbrobbel))
- chore: prepare `0.12.3` release [\#437](https://github.com/apache/arrow-rs-object-store/pull/437) ([crepererum](https://github.com/crepererum))
- aws: downgrade credential provider info! log messages to debug! [\#436](https://github.com/apache/arrow-rs-object-store/pull/436) ([asubiotto](https://github.com/asubiotto))
- feat: retry on 408 [\#426](https://github.com/apache/arrow-rs-object-store/pull/426) ([criccomini](https://github.com/criccomini))
- fix: expose source of `RetryError` [\#422](https://github.com/apache/arrow-rs-object-store/pull/422) ([crepererum](https://github.com/crepererum))
- fix\(gcp\): throw error instead of panicking if read pem fails [\#421](https://github.com/apache/arrow-rs-object-store/pull/421) ([hugocasa](https://github.com/hugocasa))
- chore: fix clippy 1.88 warnings [\#418](https://github.com/apache/arrow-rs-object-store/pull/418) ([mbrobbel](https://github.com/mbrobbel))
- Bump quick-xml to version 0.38.0 [\#417](https://github.com/apache/arrow-rs-object-store/pull/417) ([raimannma](https://github.com/raimannma))
- Prevent compilation error with all cloud features but fs turned on [\#412](https://github.com/apache/arrow-rs-object-store/pull/412) ([jder](https://github.com/jder))
- Retry requests when status code is 429 [\#410](https://github.com/apache/arrow-rs-object-store/pull/410) ([paraseba](https://github.com/paraseba))
- refactor: remove AWS dynamo integration [\#407](https://github.com/apache/arrow-rs-object-store/pull/407) ([crepererum](https://github.com/crepererum))
- refactor: `PutMultiPartOpts` =\> `PutMultiPartOptions` [\#406](https://github.com/apache/arrow-rs-object-store/pull/406) ([crepererum](https://github.com/crepererum))
- minor: Pin `tracing-attributes`, `tracing-core` to fix CI [\#404](https://github.com/apache/arrow-rs-object-store/pull/404) ([kylebarron](https://github.com/kylebarron))
- feat \(azure\): support for account in `az://` URLs [\#403](https://github.com/apache/arrow-rs-object-store/pull/403) ([ByteBaker](https://github.com/ByteBaker))
- Fix azure path parsing [\#399](https://github.com/apache/arrow-rs-object-store/pull/399) ([kylebarron](https://github.com/kylebarron))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
