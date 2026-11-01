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

# Rust Object Store

A focused, easy to use, idiomatic, high performance, `async` object
store library for interacting with object stores.

Using this crate, the same binary and code can easily run in multiple
clouds and local test environments, via a simple runtime configuration
change. Supported object stores include:

* [AWS S3](https://aws.amazon.com/s3/)
* [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/)
* [Google Cloud Storage](https://cloud.google.com/storage)
* Local files
* Memory
* [HTTP/WebDAV Storage](https://datatracker.ietf.org/doc/html/rfc2518)
* Custom implementations

Originally developed by [InfluxData](https://www.influxdata.com/) and later donated to [Apache Arrow](https://arrow.apache.org/).

See [docs.rs](https://docs.rs/object_store) for usage instructions

See [CONTRIBUTING.md] to learn how to contribute to this project.

[CONTRIBUTING.md]: https://github.com/apache/arrow-rs-object-store/blob/main/CONTRIBUTING.md

## Support for `wasm32-unknown-unknown` target

It's possible to build `object_store` for the `wasm32-unknown-unknown` target, however the cloud storage features `aws`, `azure`, `gcp`, and `http` are not supported.

```
cargo build -p object_store --target wasm32-unknown-unknown
```

## Related Apache Crates

Here are several related crates in different repositories from other Apache projects.

| Crate                    | Description                                 | Documentation                           |
| ------------------------ | ------------------------------------------- | --------------------------------------- |
| [`object_store_opendal`] | Use [`opendal`] as [`object_store`] backend | [(README)][object_store_opendal-readme] |

[`object_store_opendal`]: https://crates.io/crates/object_store_opendal
[`opendal`]: https://crates.io/crates/opendal
[object_store_opendal-readme]: https://github.com/apache/opendal/blob/main/integrations/object_store/README.md

## Community Extensions

There following community maintained crates provide additional functionality for `object_store` and are NOT governed by the Apache Software Foundation. We list them below in the hope they may be useful, but they are not official Apache projects or endorsed by the Apache Arrow project.

| Crate                        | Description                                                                      | Documentation                               |
| ---------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------- |
| [`hdfs_native_object_store`] | Use HDFS as [`object_store`] backend                                             | [(README)][hdfs_native_object_store-readme] |
| [`ic_object_store`]          | Use [ICP] blockchain as [`object_store`] backend                                 | [(README)][ic_object_store-readme]          |
| [`anda_object_store`]        | Extends the [`object_store`] with metadata management and AES-256-GCM encryption | [(README)][anda_object_store-readme]        |

[`hdfs_native_object_store`]: https://crates.io/crates/hdfs_native_object_store
[hdfs_native_object_store-readme]: https://github.com/datafusion-contrib/hdfs-native-object-store
[`ic_object_store`]: https://crates.io/crates/ic_object_store
[ic_object_store-readme]: https://github.com/ldclabs/ic-oss/tree/main/src/ic_object_store
[`anda_object_store`]: https://crates.io/crates/anda_object_store
[anda_object_store-readme]: https://github.com/ldclabs/anda-db/blob/main/rs/anda_object_store
[ICP]: https://www.internetcomputer.org/

## Release Schedule

The [`object_store`] crate follows [Semantic Versioning]. We aim to release new
versions approximately every 2 months.

Please see [the release tracker] for an up to date release schedule and to track
the progress of upcoming releases.

[`object_store`]: https://crates.io/crates/object_store
[semantic versioning]: https://semver.org/
[the release tracker]: https://github.com/apache/arrow-rs-object-store/issues/392

Planned Release Schedule

| Approximate Date | Version  | Notes                          | Ticket                                                             |
|------------------|----------|--------------------------------|:-------------------------------------------------------------------|
| July 2025        | `0.12.3` | Minor, NO breaking API changes | [#428](https://github.com/apache/arrow-rs-object-store/issues/428) |
| Sep 2025         | `0.12.4` | Minor, NO breaking API changes | [#498](https://github.com/apache/arrow-rs-object-store/issues/489) |
| TBD              | `0.13.0` | Major, breaking API changes    | [#367](https://github.com/apache/arrow-rs-object-store/issues/367) |
| TBD              | `0.13.1` | Minor, NO breaking API changes | [#393](https://github.com/apache/arrow-rs-object-store/issues/393) |
