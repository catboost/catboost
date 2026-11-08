# `boxcar`

[<img alt="crates.io" src="https://img.shields.io/crates/v/boxcar?style=for-the-badge" height="25">](https://crates.io/crates/boxcar)
[<img alt="github" src="https://img.shields.io/badge/github-boxcar-blue?style=for-the-badge" height="25">](https://github.com/ibraheemdev/boxcar)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/boxcar?style=for-the-badge" height="25">](https://docs.rs/boxcar)

A concurrent, append-only vector.

The vector provided by this crate supports lock-free `get` and `push` operations.
The vector grows internally but never reallocates, so element addresses are stable
for the lifetime of the vector. Additionally, both `get` and `push` run in constant-time.

## Examples

Appending an element to a vector and retrieving it:

```rust
let vec = boxcar::Vec::new();
let i = vec.push(42);
assert_eq!(vec[i], 42);
```

The vector can be modified by multiple threads concurrently:

```rust
let vec = boxcar::Vec::new();

// Spawn a few threads that append to the vector.
std::thread::scope(|s| for i in 0..6 {
    let vec = &vec;

    s.spawn(move || {
        // Push through the shared reference.
        vec.push(i);
    });
});

for i in 0..6 {
    assert!(vec.iter().any(|(_, &x)| x == i));
}
```

Elements can be mutated through fine-grained locking:

```rust
let vec = boxcar::Vec::new();

std::thread::scope(|s| {
    // Insert an element.
    vec.push(std::sync::Mutex::new(0));

    s.spawn(|| {
        // Mutate through the lock.
        *vec[0].lock().unwrap() += 1;
    });
});

let x = vec[0].lock().unwrap();
assert_eq!(*x, 1);
```
