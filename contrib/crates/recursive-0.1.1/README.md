# Recursive

With `recursive` you can easily make (indirectly) recursive functions without
worrying about stack overflows by marking them as `#[recursive]`:

```rust
use recursive::recursive;

#[recursive]
fn sum(nums: &[u64]) -> u64 {
    if let Some((head, tail)) = nums.split_first() {
        head + sum(tail)
    } else {
        0
    }
}
```

Functions marked with `#[recursive]` will automatically grow the stack size if
it is too small when called. See the
[crate docs](https://docs.rs/recursive/latest/) for details.

## License

`recursive` is licensed under the MIT license.
