# cpuid [![Crates.io](https://img.shields.io/crates/v/raw_cpuid.svg)](https://crates.io/crates/raw-cpuid) [![Standard checks](https://github.com/gz/rust-cpuid/actions/workflows/standard.yml/badge.svg)](https://github.com/gz/rust-cpuid/actions/workflows/standard.yml)

A library to parse the x86 CPUID instruction, written in rust with no external
dependencies. The implementation closely resembles the Intel CPUID manual
description. The library works in `no_std` environments. Some additional cargo
features require `std` (e.g., pretty printing, serialization).

- For Intel platforms: The code should be in sync with the March 2018 revision of the Intel Architectures SDM.
- For AMD platforms it should be in sync with the [AMD64 systems manual no. 24594](https://www.amd.com/system/files/TechDocs/24594.pdf),  Revision 3.32 (March 2021).

## Library usage

```rust
use raw_cpuid::CpuId;
let cpuid = CpuId::new();

if let Some(vf) = cpuid.get_vendor_info() {
    assert!(vf.as_str() == "GenuineIntel" || vf.as_str() == "AuthenticAMD");
}

let has_sse = cpuid.get_feature_info().map_or(false, |finfo| finfo.has_sse());
if has_sse {
    println!("CPU supports SSE!");
}

if let Some(cparams) = cpuid.get_cache_parameters() {
    for cache in cparams {
        let size = cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() * cache.sets();
        println!("L{}-Cache size is {}", cache.level(), size);
    }
} else {
    println!("No cache parameter information available")
}
```

## `cpuid` binary

`raw-cpuid` ships with a `cpuid` binary that can be installed to inspect the
output of the CPUID instruction on a host system.

To install, use:

```bash
cargo install raw-cpuid --features cli
```

The `cli` feature is currently required to build the binary version due to
[cargo limitations](https://github.com/rust-lang/cargo/issues/1982).

## Documentation

* [API Documentation](https://docs.rs/raw-cpuid/)
* [Examples](https://github.com/gz/rust-cpuid/tree/master/examples)
