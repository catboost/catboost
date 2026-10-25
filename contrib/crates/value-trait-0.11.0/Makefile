wasmtest:
	cargo build --tests --target wasm32-wasi --target-dir target
	wasmtime run --wasm-features=simd target/wasm32-wasi/debug/deps/value_trait*.wasm