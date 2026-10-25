# unicode-reverse

Unicode-aware in-place string reversal for Rust UTF-8 strings.

* [Documentation](https://docs.rs/unicode-reverse)
* [crates.io](https://crates.io/crates/unicode-reverse)
* [Release notes](https://github.com/mbrubeck/unicode-reverse/blob/master/CHANGELOG.md)

The [`reverse_grapheme_clusters_in_place`][0] function reverses a string slice in-place without
allocating any memory on the heap.  It correctly handles multi-byte UTF-8 sequences and
grapheme clusters, including combining marks and astral characters such as Emoji.

## Example

```rust
use unicode_reverse::reverse_grapheme_clusters_in_place;

let mut x = "man\u{0303}ana".to_string();
println!("{}", x); // prints "mañana"

reverse_grapheme_clusters_in_place(&mut x);
println!("{}", x); // prints "anañam"
```

## Background

As described in [this article by Mathias Bynens][1], naively reversing a Unicode string can go
wrong in several ways. For example, merely reversing the `chars` (Unicode Scalar Values) in a
string can cause combining marks to become attached to the wrong characters:

```rust
let x = "man\u{0303}ana";
println!("{}", x); // prints "mañana"

let y: String = x.chars().rev().collect();
println!("{}", y); // prints "anãnam": Oops! The '~' is now applied to the 'a'.
```

Reversing the [grapheme clusters][2] of the string fixes this problem:

```rust
extern crate unicode_segmentation;
use unicode_segmentation::UnicodeSegmentation;

fn main() {
    let x = "man\u{0303}ana";
    let y: String = x.graphemes(true).rev().collect();
    println!("{}", y); // prints "anañam"
}
```

The `reverse_grapheme_clusters_in_place` function from this crate performs this same operation,
but performs the reversal in-place rather than allocating a new string.

Note: Even grapheme-level reversal may produce unexpected output if the input string contains
certain non-printable control codes, such as directional formatting characters. Handling such
characters is outside the scope of this crate.

## Algorithm

The implementation is very simple. It makes two passes over the string's contents:

1. For each grapheme cluster, reverse the bytes within the grapheme cluster in-place.
2. Reverse the bytes of the entire string in-place.

After the second pass, each grapheme cluster has been reversed twice, so its bytes are now back
in their original order, but the clusters are now in the opposite order within the string.

## no_std

This crate does not depend on libstd, so it can be used in [`no_std` projects][3].

[0]: https://docs.rs/unicode-reverse/*/unicode_reverse/fn.reverse_grapheme_clusters_in_place.html
[1]: https://mathiasbynens.be/notes/javascript-unicode
[2]: http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries
[3]: https://doc.rust-lang.org/book/no-stdlib.html
