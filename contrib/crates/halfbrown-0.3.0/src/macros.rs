#![warn(missing_docs)]
#![warn(unused_results)]

//! Taken from: <https://docs.rs/maplit/1/>
//! Macros for halfbrown literals.
//!
//! ```no_run
//! #[macro_use] extern crate halfbrown;
//!
//! # fn main() {
//! let map = hashmap!{
//!     "a" => 1,
//!     "b" => 2,
//! };
//! # }
//! ```
//!
//! The **halfbrown** crate uses `=>` syntax to separate the key and value for the
//! mapping macros. (It was not possible to use `:` as separator due to syntactic
//! restrictions in regular `macro_rules!` macros.)
//!
//! Note that rust macros are flexible in which brackets you use for the invocation.
//! You can use them as `hashmap!{}` or `hashmap![]` or `hashmap!()`.
//!
//! Generic container macros already exist elsewhere, so those are not provided
//! here at the moment.

#[macro_export]
/// Create a **`HashMap`** from a list of key-value pairs
///
/// ## Example
///
/// ```no_run
/// #[macro_use] extern crate halfbrown;
/// # fn main() {
///
/// let map = hashmap!{
///     "a" => 1,
///     "b" => 2,
/// };
/// assert_eq!(map["a"], 1);
/// assert_eq!(map["b"], 2);
/// assert_eq!(map.get("c"), None);
/// # }
/// ```
macro_rules! hashmap {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(hashmap!(@single $rest)),*]));

    ($($key:expr => $value:expr,)+) => { hashmap!($($key => $value),+) };
    ($($key:expr => $value:expr),*) => {
        {
            let _cap = hashmap!(@count $($key),*);
            #[allow(clippy::let_and_return)]
            let mut _map: $crate::HashMap<_,_> = $crate::HashMap::with_capacity(_cap);
            $(
                #[allow(let_underscore_drop)]
                let _: Option<_> = _map.insert($key, $value);
            )*
            _map
        }
    };
}

/// Identity function. Used as the fallback for conversion.
#[doc(hidden)]
pub fn __id<T>(t: T) -> T {
    t
}

#[test]
fn test_hashmap() {
    use crate::SizedHashMap;
    let names = hashmap! {
        1 => "one",
        2 => "two",
    };
    assert_eq!(names.len(), 2);
    assert_eq!(names[&1], "one");
    assert_eq!(names[&2], "two");
    assert_eq!(names.get(&3), None);

    let empty: SizedHashMap<i32, i32> = hashmap! {};
    assert_eq!(empty.len(), 0);

    let _nested_compiles = hashmap! {
        1 => hashmap!{0 => 1 + 2,},
        2 => hashmap!{1 => 1,},
    };
}
