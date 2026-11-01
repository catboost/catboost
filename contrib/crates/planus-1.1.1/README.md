# Planus – alternative flatbuffer implementation

[![Build Status](https://github.com/planus-org/planus/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/planus-org/planus/actions/workflows/rust.yml)
![Rustc Version 1.70+](https://img.shields.io/badge/rustc-1.70+-lightgray.svg)
[![License](https://img.shields.io/crates/l/planus)](https://crates.io/crates/planus)
[![Crates.io](https://img.shields.io/crates/v/planus)](https://crates.io/crates/planus)
[![Rust Documentation](https://img.shields.io/badge/api-rustdoc-blue.svg)](https://docs.rs/planus)
[![Discord](https://img.shields.io/discord/1011185439270510665.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/M5uJswg6Xd)

Planus is an alternative compiler for [flatbuffers](https://google.github.io/flatbuffers/), an efficient cross platform serialization library.

## Getting started

First, install the command line utility

```shell
cargo install planus-cli
```

Next, write a flatbuffers file (or use an existing one). Then you can generate code using the command

```shell
planus rust -o <output_path.rs> <input_file.fbs>
```

For a complete example, see [examples/rust](examples/rust).

## Features

- `planus view`: a TUI viewer for serialized flatbuffer files

![Planus View](planus-view.png)

To use it you need to specify the `.fbs` file, the root type and the binary file:

```shell
planus view test/rust/test_files/alignment.fbs Root test/rust/test_files/alignment/serialize/alignment.bin
```

- `planus format`: formatter for `.fbs` files

- `planus check`: check validity of `.fbs` files with good error messages

- `planus dot`: generate a DOT graph for `.fbs` files

- `planus rust`: generate Rust bindings for `.fbs` files

## Goals

- **User experience**: Our command-line interface has excellent error and help messages. We aim to output good error messages even for non-supported features.
- **Idiomatic code**: We want to generate highly idiomatic code that should be very familiar to programmers of the target programming language.
- **Safety**: Any undefined/unsafe behavior in the generated code is considered a critical bug.
- **Performance**: We want to be at least as performant as the official implementation.
- **Modularity**: We have written our code such that parsing, validation and translation are clearly separated. We hope this will make it easy to implement additional backends with full support.
- **Opinionated**: We are in some cases more strict than the official implementation.
- **Developer tools**: We want to build good developer tools using our library. This includes at least a schema formatter and a tool to output a [DOT graph](<https://en.wikipedia.org/wiki/DOT_(graph_description_language)>) of the types in a schema.
- **Rust**: By choosing to use Rust for our compiler, we are able to utilize excellent crates such as [clap](https://github.com/clap-rs/clap), [LALRPOP](https://github.com/lalrpop/lalrpop) and [codespan](https://github.com/brendanzab/codespan) as force multipliers.

## Non-goals

- **Full feature parity**: Certain features are difficult to re-implement in a clean fashion.
- **API-level compatibility**: We aim for binary-level compatibility, but the code we generate will not works directly with the APIs of the official implementation.
- **Language-specific extensions**: We do not plan to support any extensions that break support between different languages.
- **Validation-free access**: Validation-free access makes it very easy to shoot yourself in the foot. We will not provide such APIs.

## Languages supported

Currently we only support Rust, though we plan to add support for more languages in the future. Pull requests are welcome!

## Community

Before contributing or participating in discussions with the community, you should familiarize yourself with the [**Code of Conduct**](https://www.rust-lang.org/policies/code-of-conduct) we follow in this project.

- [**Discord**](https://discord.gg/M5uJswg6Xd): Planus' official discord server.
- [**GitHub Discussions**](https://github.com/planus-org/planus/discussions): The best place for complex discussions about planus.

If you would like to contribute, please open an issue or pull request to discuss your idea.

For more more open-ended questions or complex decisions, feel free to open a Github Discussion or ping the developers on Discord.

## Status of the implementation

We support most of the base language, though some parts have not been tested in-depth.

Things we do not currently support:

- `rpc_service`
- `file_extension`, `file_identifier` and `root_type`
- Fixed-size arrays
- Any attribute besides `required`, `deprecated`, `id` or `force_align`.
- Some of the more exotic literal values, like hexadecimal floats or unicode surrogate pair parsing.

Things we will probably never support:

- `native_includes`
- More than one `namespace` per file.
- Flexbuffers.

## Minimum Supported Rust Version (MSRV)

Our current Minimum Supported Rust Version is 1.75.0. When adding features, we will follow these guidelines:

- We will aim to support the latest five minor Rust versions. This gives you a 6 month window to upgrade your compiler.
- Any change to the MSRV will be accompanied with a minor version bump

## I think I found a bug!

Please file an issue! We also consider poor error messages or unintuitive behavior to be high-priority bugs.
