snap
====
A pure Rust implementation of the
[Snappy compression algorithm](https://google.github.io/snappy/).
Includes streaming compression and decompression using the Snappy frame format.
This implementation is ported from both the
[reference C++ implementation](https://github.com/google/snappy)
and the
[Go implementation](https://github.com/golang/snappy).

[![Build status](https://github.com/BurntSushi/rust-snappy/workflows/ci/badge.svg)](https://github.com/BurntSushi/rust-snappy/actions)
[![](https://meritbadge.herokuapp.com/snap)](https://crates.io/crates/snap)

Licensed under the BSD 3-Clause.


### Documentation

https://docs.rs/snap


### Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
snap = "1"
```


### Example: compress data on `stdin`

This program reads data from `stdin`, compresses it and emits it to `stdout`.
This example can be found in `examples/compress.rs`:

```rust
use std::io;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    let mut rdr = stdin.lock();
    // Wrap the stdout writer in a Snappy writer.
    let mut wtr = snap::write::FrameEncoder::new(stdout.lock());
    io::copy(&mut rdr, &mut wtr).expect("I/O operation failed");
}
```


### Example: decompress data on `stdin`

This program reads data from `stdin`, decompresses it and emits it to `stdout`.
This example can be found in `examples/decompress.rs`:

```rust
use std::io;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    // Wrap the stdin reader in a Snappy reader.
    let mut rdr = snap::read::FrameDecoder::new(stdin.lock());
    let mut wtr = stdout.lock();
    io::copy(&mut rdr, &mut wtr).expect("I/O operation failed");
}
```


### Example: the szip tool

`szip` is a tool with similar behavior as `gzip`, except it uses Snappy
compression. It can be installed with Cargo:

```
$ cargo install szip
```

To compress a file, run `szip file`. To decompress a file, run
`szip -d file.sz`. See `szip --help` for more details.


### Testing

This crate is tested against the reference C++ implementation of Snappy.
Currently, compression is byte-for-byte equivalent with the C++ implementation.
This seems like a reasonable starting point, although it is not necessarily
a goal to always maintain byte-for-byte equivalence.

Tests against the reference C++ implementation can be run with
`cargo test --features cpp`. Note that you will need to have the C++ Snappy
library in your `LD_LIBRARY_PATH` (or equivalent).

To run tests, you'll need to explicitly run the `test` crate:

```
$ cargo test --manifest-path test/Cargo.toml
```

To test that this library matches the output of the reference C++ library, use:

```
$ cargo test --manifest-path test/Cargo.toml --features cpp
```

Tests are in a separate crate because of the dependency on the C++ reference
library. Namely, Cargo does not yet permit optional dev dependencies.


### Minimum Rust version policy

This crate's minimum supported `rustc` version is `1.39.0`.

The current policy is that the minimum Rust version required to use this crate
can be increased in minor version updates. For example, if `crate 1.0` requires
Rust 1.20.0, then `crate 1.0.z` for all values of `z` will also require Rust
1.20.0 or newer. However, `crate 1.y` for `y > 0` may require a newer minimum
version of Rust.

In general, this crate will be conservative with respect to the minimum
supported version of Rust.


### Performance

The performance of this implementation should roughly match the performance of
the C++ implementation on x86_64. Below are the results of the microbenchmarks
(as defined in the C++ library):

```
group                         snappy/cpp/                            snappy/snap/
-----                         -----------                            ------------
compress/zflat00_html         1.00     94.5±0.62µs  1033.1 MB/sec    1.02     96.1±0.74µs  1016.2 MB/sec
compress/zflat01_urls         1.00   1182.3±8.89µs   566.3 MB/sec    1.04  1235.3±11.99µs   542.0 MB/sec
compress/zflat02_jpg          1.00      7.2±0.11µs    15.9 GB/sec    1.01      7.3±0.06µs    15.8 GB/sec
compress/zflat03_jpg_200      1.10    262.4±1.84ns   727.0 MB/sec    1.00    237.5±2.95ns   803.2 MB/sec
compress/zflat04_pdf          1.02     10.3±0.18µs     9.2 GB/sec    1.00     10.1±0.16µs     9.4 GB/sec
compress/zflat05_html4        1.00    399.2±5.36µs   978.4 MB/sec    1.01    404.0±2.46µs   966.8 MB/sec
compress/zflat06_txt1         1.00    397.3±2.61µs   365.1 MB/sec    1.00    398.5±3.06µs   364.0 MB/sec
compress/zflat07_txt2         1.00    352.8±3.20µs   338.4 MB/sec    1.01    355.2±5.01µs   336.1 MB/sec
compress/zflat08_txt3         1.01   1058.8±6.85µs   384.4 MB/sec    1.00   1051.8±6.74µs   386.9 MB/sec
compress/zflat09_txt4         1.00   1444.1±8.10µs   318.2 MB/sec    1.00  1450.0±13.36µs   316.9 MB/sec
compress/zflat10_pb           1.00     85.1±0.58µs  1328.6 MB/sec    1.02     87.0±0.90µs  1300.2 MB/sec
compress/zflat11_gaviota      1.07    311.9±4.27µs   563.5 MB/sec    1.00    291.9±1.86µs   602.3 MB/sec
decompress/uflat00_html       1.03     36.9±0.28µs     2.6 GB/sec    1.00     36.0±0.25µs     2.7 GB/sec
decompress/uflat01_urls       1.04    437.4±2.89µs  1530.7 MB/sec    1.00    419.9±3.10µs  1594.6 MB/sec
decompress/uflat02_jpg        1.00      4.6±0.05µs    24.9 GB/sec    1.00      4.6±0.03µs    25.0 GB/sec
decompress/uflat03_jpg_200    1.08    122.4±1.06ns  1558.6 MB/sec    1.00    112.8±1.35ns  1690.8 MB/sec
decompress/uflat04_pdf        1.00      5.7±0.05µs    16.8 GB/sec    1.10      6.2±0.07µs    15.3 GB/sec
decompress/uflat05_html4      1.01    164.1±1.71µs     2.3 GB/sec    1.00    162.6±2.16µs     2.3 GB/sec
decompress/uflat06_txt1       1.08    146.6±1.01µs   989.5 MB/sec    1.00    135.3±1.11µs  1072.0 MB/sec
decompress/uflat07_txt2       1.09    130.2±0.93µs   916.6 MB/sec    1.00    119.2±0.96µs  1001.8 MB/sec
decompress/uflat08_txt3       1.07    387.2±2.30µs  1051.0 MB/sec    1.00    361.9±6.29µs  1124.7 MB/sec
decompress/uflat09_txt4       1.09    536.1±3.47µs   857.2 MB/sec    1.00    494.0±5.05µs   930.2 MB/sec
decompress/uflat10_pb         1.00     32.5±0.19µs     3.4 GB/sec    1.05     34.0±0.48µs     3.2 GB/sec
decompress/uflat11_gaviota    1.00    142.1±2.05µs  1236.7 MB/sec    1.00    141.5±0.92µs  1242.3 MB/sec
```

Notes: These benchmarks were run with Snappy/C++ 1.1.8. Both the C++ and Rust
benchmarks were run with the same benchmark harness. Benchmarks were run on an
Intel i7-6900K.

Additionally, here are the benchmarks run on the same machine from the Go
implementation of Snappy (which has a hand rolled implementation in Assembly).
Note that these were run using Go's microbenchmark tool, so the numbers may not
be directly comparable, but they should serve as a useful signpost:

```
Benchmark_UFlat0           25040             45180 ns/op        2266.49 MB/s
Benchmark_UFlat1            2648            451475 ns/op        1555.10 MB/s
Benchmark_UFlat2          229965              4788 ns/op        25709.01 MB/s
Benchmark_UFlat3        11355555               101 ns/op        1973.65 MB/s
Benchmark_UFlat4          196551              6055 ns/op        16912.64 MB/s
Benchmark_UFlat5            6016            189219 ns/op        2164.68 MB/s
Benchmark_UFlat6            6914            166371 ns/op         914.16 MB/s
Benchmark_UFlat7            8173            142506 ns/op         878.41 MB/s
Benchmark_UFlat8            2744            436424 ns/op         977.84 MB/s
Benchmark_UFlat9            1999            591141 ns/op         815.14 MB/s
Benchmark_UFlat10          28885             37291 ns/op        3180.04 MB/s
Benchmark_UFlat11           7308            163366 ns/op        1128.26 MB/s
Benchmark_ZFlat0           12902             91231 ns/op        1122.43 MB/s
Benchmark_ZFlat1             997           1200579 ns/op         584.79 MB/s
Benchmark_ZFlat2          136762              7832 ns/op        15716.53 MB/s
Benchmark_ZFlat3         4896124               245 ns/op         817.27 MB/s
Benchmark_ZFlat4          117643             10129 ns/op        10109.44 MB/s
Benchmark_ZFlat5            2934            394742 ns/op        1037.64 MB/s
Benchmark_ZFlat6            3008            382877 ns/op         397.23 MB/s
Benchmark_ZFlat7            3411            344916 ns/op         362.93 MB/s
Benchmark_ZFlat8             966           1057985 ns/op         403.36 MB/s
Benchmark_ZFlat9             854           1429024 ns/op         337.20 MB/s
Benchmark_ZFlat10          13861             83040 ns/op        1428.08 MB/s
Benchmark_ZFlat11           4070            293952 ns/op         627.04 MB/s
```

To run benchmarks, including the reference C++ implementation, do the
following:

```
$ cd bench
$ cargo bench --features cpp -- --save-baseline snappy
```

To compare them, as shown above, install
[`critcmp`](https://github.com/BurntSushi/critcmp)
and run (assuming you saved the baseline above under the name `snappy`):

```
$ critcmp snappy -g '.*?/(.*$)'
```

Finally, the Go benchmarks were run with the following command on commit
`ff6b7dc8`:

```
$ go test -cpu 1 -bench Flat -download
```


### Comparison with other Snappy crates

* `snappy` - These are bindings to the C++ library. No support for the Snappy
  frame format.
* `snappy_framed` - Implements the Snappy frame format on top of the `snappy`
  crate.
* `rsnappy` - Written in pure Rust, but lacks documentation and the Snappy
  frame format. Performance is unclear and tests appear incomplete.
* `snzip` - Was created and immediately yanked from crates.io.
