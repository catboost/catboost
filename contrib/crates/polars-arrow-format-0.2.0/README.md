# Arrow generated IPC format

Generated flatbuffers code for Rust using [planus](https://crates.io/crates/planus).

* Planus version 1.1.1
* Apache Arrow version 20.0.0

## Cargo features

This package is divided in 3 features:

* `ipc` for Apache [IPC format](https://github.com/apache/arrow/tree/master/format)
* `flight-data` for the `prost` part of the Apache [Flight protocol](https://github.com/apache/arrow/blob/master/format/Flight.proto)
* `flight-service` for the `tonic` part of the Apache [Flight protocol](https://github.com/apache/arrow/blob/master/format/Flight.proto)
