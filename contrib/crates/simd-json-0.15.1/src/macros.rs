/// Construct a `simd_json::Value` from a JSON literal.
///
/// Taken from: <https://github.com/serde-rs/json/blob/5b5f95831d9e0d769367b30b76a686339bffd209/src/macros.rs>
///
/// ```edition2018
/// # use simd_json::json;
/// #
/// let value = json!({
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     }
/// });
/// ```
///
/// Variables or expressions can be interpolated into the JSON literal. Any type
/// interpolated into an array element or object value must implement Serde's
/// `Serialize` trait, while any type interpolated into a object key must
/// implement `Into<String>`. If the `Serialize` implementation of the
/// interpolated type decides to fail, or if the interpolated type contains a
/// map with non-string keys, the `json!` macro will panic.
///
/// ```edition2018
/// # use simd_json::json;
/// #
/// let code = 200;
/// let features = vec!["serde", "json"];
///
/// let value = json!({
///     "code": code,
///     "success": code == 200,
///     "payload": {
///         features[0]: features[1]
///     }
/// });
/// ```
///
/// Trailing commas are allowed inside both arrays and objects.
///
/// ```edition2018
/// # use simd_json::json;
/// #
/// let value = json!([
///     "notice",
///     "the",
///     "trailing",
///     "comma -->",
/// ]);
/// ```
///
/// It works for both Borrowed and owned values when used with `.into()`
/// ```edition2018
/// # use simd_json::{json, OwnedValue, BorrowedValue};
/// #
/// let owned_value: OwnedValue = json!({
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     }
/// });
///
/// let borrowed_value: BorrowedValue = json!({
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     },
///     "empty_obj": {},
///     "empty_array": [],
/// }).into();
/// ````
#[macro_export(local_inner_macros)]
macro_rules! json {
    // Hide distracting implementation details from the generated rustdoc.
    ($($json:tt)+) => {
        json_internal_owned!($($json)+)
    };
}

/// Constructs a `simd_json::Value` from a JSON literal and allows specifying whether it generates
/// an owned or borrowed variant.
///
/// Adapted from: <https://github.com/serde-rs/json/blob/5b5f95831d9e0d769367b30b76a686339bffd209/src/macros.rs>
///
/// Create an owned value of the form:
///
/// ```edition2018
/// # use simd_json::{json_typed, OwnedValue};
/// #
/// let value: OwnedValue = json_typed!(owned, {
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     }
/// });
/// ```
///
/// Create a borrowed value of the form:
///
/// ```edition2018
/// # use simd_json::{json_typed, BorrowedValue};
/// #
/// let value: BorrowedValue = json_typed!(borrowed, {
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     }
/// });
/// ```
///
/// Variables or expressions can be interpolated into the JSON literal. Any type
/// interpolated into an array element or object value must implement Serde's
/// `Serialize` trait, while any type interpolated into a object key must
/// implement `Into<String>`. If the `Serialize` implementation of the
/// interpolated type decides to fail, or if the interpolated type contains a
/// map with non-string keys, the `json!` macro will panic.
///
/// ```edition2018
/// # use simd_json::json_typed;
/// #
/// let code = 200;
/// let features = vec!["serde", "json"];
///
/// let value = json_typed!(owned, {
///     "code": code,
///     "success": code == 200,
///     "payload": {
///         features[0]: features[1]
///     }
/// });
/// ```
///
/// Trailing commas are allowed inside both arrays and objects.
///
/// ```edition2018
/// # use simd_json::json_typed;
/// #
/// let value = json_typed!(borrowed, [
///     "notice",
///     "the",
///     "trailing",
///     "comma -->",
/// ]);
/// ```
///
/// It works for both Borrowed and owned values natively without the
/// need for calling `into()` unlike the `json!` macro which supports
/// owned values only.
///
/// ```edition2018
/// # use simd_json::{json_typed, OwnedValue, BorrowedValue};
/// #
/// let owned_value: OwnedValue = json_typed!(owned, {
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     }
/// });
///
/// let borrowed_value: BorrowedValue = json_typed!(borrowed, {
///     "code": 200,
///     "success": true,
///     "payload": {
///         "features": [
///             "serde",
///             "json"
///         ]
///     },
///     "empty_obj": {},
///     "empty_array": [],
/// }).into();
/// ````
#[macro_export(local_inner_macros)]
macro_rules! json_typed {
    (owned, $($json:tt)+) => {
        json_internal_owned!($($json)+)
    };
    (borrowed, $($json:tt)+) => {
        json_internal_borrowed!($($json)+)
   };

}

// Rocket relies on this because they export their own `json!` with a different
// doc comment than ours, and various Rust bugs prevent them from calling our
// `json!` from their `json!` so they call `json_internal!` directly. Check with
// @SergioBenitez before making breaking changes to this macro.
//
// Changes are fine as long as `json_internal!` does not call any new helper
// macros and can still be invoked as `json_internal!($($json)+)`.
#[cfg(feature = "serde_impl")]
#[macro_export(local_inner_macros)]
#[doc(hidden)]
macro_rules! json_internal_owned {
    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an array [...]. Produces a vec![...]
    // of the elements.
    //
    // Must be invoked as: json_internal_owned!(@array [] $($tt)*)
    //////////////////////////////////////////////////////////////////////////

    // Done with trailing comma.
    (@array [$($elems:expr_2021,)*]) => {
        json_internal_vec![$($elems,)*]
    };

    // Done without trailing comma.
    (@array [$($elems:expr_2021),*]) => {
        json_internal_vec![$($elems),*]
    };

    // Next element is `null`.
    (@array [$($elems:expr_2021,)*] null $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!(null)] $($rest)*)
    };

    // Next element is `true`.
    (@array [$($elems:expr_2021,)*] true $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!(true)] $($rest)*)
    };

    // Next element is `false`.
    (@array [$($elems:expr_2021,)*] false $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!(false)] $($rest)*)
    };

    // Next element is an array.
    (@array [$($elems:expr_2021,)*] [$($array:tt)*] $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!([$($array)*])] $($rest)*)
    };

    // Next element is a map.
    (@array [$($elems:expr_2021,)*] {$($map:tt)*} $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!({$($map)*})] $($rest)*)
    };

    // Next element is an expression followed by comma.
    (@array [$($elems:expr_2021,)*] $next:expr_2021, $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!($next),] $($rest)*)
    };

    // Last element is an expression with no trailing comma.
    (@array [$($elems:expr_2021,)*] $last:expr_2021) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!($last)])
    };

    // Comma after the most recent element.
    (@array [$($elems:expr_2021),*] , $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)*] $($rest)*)
    };

    // Unexpected token after most recent element.
    (@array [$($elems:expr_2021),*] $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected)
    };

    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an object {...}. Each entry is
    // inserted into the given map variable. Entries are parsed from the TT
    // and then inserted into a stack of entries to be appended all at once
    // at the end, allowing us to pre-allocate the map.
    //
    // Must be invoked as: json_internal!(@object $map () ($($tt)*) ($($tt)*))
    //
    // We require two copies of the input tokens so that we can match on one
    // copy and trigger errors on the other copy.
    //////////////////////////////////////////////////////////////////////////

    // Done.
    (@object $object:ident [@entries] () () ()) => {};

    // Count the number of entries to insert.
    // Modified version of https://docs.rs/halfbrown/0.1.11/src/halfbrown/macros.rs.html#46-62
    (@object @count [@entries $(($value:expr_2021 => $($key:tt)+))*]) => {
        <[()]>::len(&[$(json_internal_owned!(@object @count @single ($($key)+))),*])
    };
    (@object @count @single ($($key:tt)+)) => {()};

    // Done. Insert all entries from the stack
    (@object $object:ident [@entries $(($value:expr_2021 => $($key:tt)+))*] () () ()) => {
        let len = json_internal_owned!(@object @count [@entries $(($value => $($key)+))*]);
        $object = $crate::value::owned::Object::with_capacity_and_hasher(len, $crate::value::ObjectHasher::default());
        $(
            #[allow(let_underscore_drop)]
            let _: Option<_> = $object.insert(($($key)+).into(), $value);
        )*
    };

    // Insert the current entry (followed by trailing comma) into the stack.
    // Entries are inserted in reverse order, the captured $entries is expanded first,
    // keeping inserts in the same order that they're defined in the macro.
    //
    // We expand the $key _after_ the $value since $key => $value leads to a parsing ambiguity.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr_2021) , $($rest:tt)*) => {
        json_internal_owned!(@object $object [@entries $($entries)* ($value => $($key)+) ] () ($($rest)*) ($($rest)*));
    };

    // Insert the last entry (without trailing comma) into the stack.
    // At this point the recursion is complete and the entries can now be inserted.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr_2021)) => {
        json_internal_owned!(@object $object [@entries $($entries)* ($value => $($key)+) ] () () ());
    };

    // Current entry followed by unexpected token.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr_2021) $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected);
    };


    // Next value is `null`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!(null)) $($rest)*);
    };

    // Next value is `true`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!(true)) $($rest)*);
    };

    // Next value is `false`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!(false)) $($rest)*);
    };

    // Next value is an array.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!([$($array)*])) $($rest)*);
    };

    // Next value is a map.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!({$($map)*})) $($rest)*);
    };

    // Next value is an expression followed by comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr_2021 , $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!($value)) , $($rest)*);
    };

    // Last value is an expression with no trailing comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr_2021) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!($value)));
    };

    // Missing value for last entry. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (:) $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_owned!();
    };

    // Missing colon and value for last entry. Trigger a reasonable error
    // message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) () $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_owned!();
    };

    // Misplaced colon. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] () (: $($rest:tt)*) ($colon:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `:`".
        json_unexpected!($colon);
    };

    // Found a comma inside a key. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) (, $($rest:tt)*) ($comma:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `,`".
        json_unexpected!($comma);
    };

    // Key is fully parenthesized. This avoids clippy double_parens false
    // positives because the parenthesization may be necessary here.
    (@object $object:ident [@entries $($entries:tt)*] () (($key:expr_2021) : $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] ($key) (: $($rest)*) (: $($rest)*));
    };

    // Munch a token into the current key.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    //////////////////////////////////////////////////////////////////////////
    // The main implementation.
    //
    // Must be invoked as: json_internal!($($json)+)
    //////////////////////////////////////////////////////////////////////////

    (null) => {
        $crate::value::owned::Value::Static($crate::StaticNode::Null)
    };

    (true) => {
        $crate::value::owned::Value::Static($crate::StaticNode::Bool(true))
    };

    (false) => {
        $crate::value::owned::Value::Static($crate::StaticNode::Bool(false))
    };

    ([]) => {
        $crate::value::owned::Value::Array(Box::new(json_internal_vec![]))
    };

    ([ $($tt:tt)+ ]) => {
        $crate::value::owned::Value::Array(Box::new(json_internal_owned!(@array [] $($tt)+)))
    };

    ({}) => {
        {
            use $crate::value::ValueBuilder;
            $crate::value::owned::Value::object()
        }
    };

    ({ $($tt:tt)+ }) => {
        $crate::value::owned::Value::from({
            let mut object;
            json_internal_owned!(@object object [@entries] () ($($tt)+) ($($tt)+));
            object
        })
    };

    // Any Serialize type: numbers, strings, struct literals, variables etc.
    // Must be below every other rule.
    ($other:expr_2021) => {
        $crate::serde::to_owned_value(&$other).expect("serde::to_owned_value")
    };
}

#[cfg(not(feature = "serde_impl"))]
#[macro_export(local_inner_macros)]
#[doc(hidden)]
macro_rules! json_internal_owned {
    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an array [...]. Produces a vec![...]
    // of the elements.
    //
    // Must be invoked as: json_internal_owned!(@array [] $($tt)*)
    //////////////////////////////////////////////////////////////////////////

    // Done with trailing comma.
    (@array [$($elems:expr,)*]) => {
        json_internal_vec![$($elems,)*]
    };

    // Done without trailing comma.
    (@array [$($elems:expr),*]) => {
        json_internal_vec![$($elems),*]
    };

    // Next element is `null`.
    (@array [$($elems:expr,)*] null $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!(null)] $($rest)*)
    };

    // Next element is `true`.
    (@array [$($elems:expr,)*] true $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!(true)] $($rest)*)
    };

    // Next element is `false`.
    (@array [$($elems:expr,)*] false $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!(false)] $($rest)*)
    };

    // Next element is an array.
    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!([$($array)*])] $($rest)*)
    };

    // Next element is a map.
    (@array [$($elems:expr,)*] {$($map:tt)*} $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!({$($map)*})] $($rest)*)
    };

    // Next element is an expression followed by comma.
    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!($next),] $($rest)*)
    };

    // Last element is an expression with no trailing comma.
    (@array [$($elems:expr,)*] $last:expr) => {
        json_internal_owned!(@array [$($elems,)* json_internal_owned!($last)])
    };

    // Comma after the most recent element.
    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        json_internal_owned!(@array [$($elems,)*] $($rest)*)
    };

    // Unexpected token after most recent element.
    (@array [$($elems:expr),*] $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected)
    };

    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an object {...}. Each entry is
    // inserted into the given map variable. Entries are parsed from the TT
    // and then inserted into a stack of entries to be appended all at once
    // at the end, allowing us to pre-allocate the map.
    //
    // Must be invoked as: json_internal!(@object $map () ($($tt)*) ($($tt)*))
    //
    // We require two copies of the input tokens so that we can match on one
    // copy and trigger errors on the other copy.
    //////////////////////////////////////////////////////////////////////////

    // Done.
    (@object $object:ident [@entries] () () ()) => {};

    // Count the number of entries to insert.
    // Modified version of https://docs.rs/halfbrown/0.1.11/src/halfbrown/macros.rs.html#46-62
    (@object @count [@entries $(($value:expr => $($key:tt)+))*]) => {
        <[()]>::len(&[$(json_internal_owned!(@object @count @single ($($key)+))),*])
    };
    (@object @count @single ($($key:tt)+)) => {()};

    // Done. Insert all entries from the stack
    (@object $object:ident [@entries $(($value:expr => $($key:tt)+))*] () () ()) => {
        let len = json_internal_owned!(@object @count [@entries $(($value => $($key)+))*]);
        $object = $crate::value::owned::Object::with_capacity_and_hasher(len, $crate::value::ObjectHasher::default());
        $(
            #[allow(clippy::let_underscore_drop)]
            let _: Option<_> = $object.insert(($($key)+).into(), $value);
        )*
    };

    // Insert the current entry (followed by trailing comma) into the stack.
    // Entries are inserted in reverse order, the captured $entries is expanded first,
    // keeping inserts in the same order that they're defined in the macro.
    //
    // We expand the $key _after_ the $value since $key => $value leads to a parsing ambiguity.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        json_internal_owned!(@object $object [@entries $($entries)* ($value => $($key)+) ] () ($($rest)*) ($($rest)*));
    };

    // Insert the last entry (without trailing comma) into the stack.
    // At this point the recursion is complete and the entries can now be inserted.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr)) => {
        json_internal_owned!(@object $object [@entries $($entries)* ($value => $($key)+) ] () () ());
    };

    // Current entry followed by unexpected token.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr) $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected);
    };


    // Next value is `null`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!(null)) $($rest)*);
    };

    // Next value is `true`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!(true)) $($rest)*);
    };

    // Next value is `false`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!(false)) $($rest)*);
    };

    // Next value is an array.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!([$($array)*])) $($rest)*);
    };

    // Next value is a map.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!({$($map)*})) $($rest)*);
    };

    // Next value is an expression followed by comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr , $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!($value)) , $($rest)*);
    };

    // Last value is an expression with no trailing comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_owned!($value)));
    };

    // Missing value for last entry. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (:) $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_owned!();
    };

    // Missing colon and value for last entry. Trigger a reasonable error
    // message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) () $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_owned!();
    };

    // Misplaced colon. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] () (: $($rest:tt)*) ($colon:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `:`".
        json_unexpected!($colon);
    };

    // Found a comma inside a key. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) (, $($rest:tt)*) ($comma:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `,`".
        json_unexpected!($comma);
    };

    // Key is fully parenthesized. This avoids clippy double_parens false
    // positives because the parenthesization may be necessary here.
    (@object $object:ident [@entries $($entries:tt)*] () (($key:expr) : $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] ($key) (: $($rest)*) (: $($rest)*));
    };

    // Munch a token into the current key.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        json_internal_owned!(@object $object [@entries $($entries)*] ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    //////////////////////////////////////////////////////////////////////////
    // The main implementation.
    //
    // Must be invoked as: json_internal!($($json)+)
    //////////////////////////////////////////////////////////////////////////

    (null) => {
        $crate::value::owned::Value::Static($crate::StaticNode::Null)
    };

    (true) => {
        $crate::value::owned::Value::Static($crate::StaticNode::Bool(true))
    };

    (false) => {
        $crate::value::owned::Value::Static($crate::StaticNode::Bool(false))
    };

    ([]) => {
        $crate::value::owned::Value::Array(Box::new(json_internal_vec![]))
    };

    ([ $($tt:tt)+ ]) => {
        $crate::value::owned::Value::Array(Box::new(json_internal_owned!(@array [] $($tt)+)))
    };

    ({}) => {
        {
            use $crate::value::ValueBuilder;
            $crate::value::owned::Value::object()
        }
    };

    ({ $($tt:tt)+ }) => {
        $crate::value::owned::Value::from({
            let mut object;
            json_internal_owned!(@object object [@entries] () ($($tt)+) ($($tt)+));
            object
        })
    };

    // Any Serialize type: numbers, strings, struct literals, variables etc.
    // Must be below every other rule.
    ($other:expr) => {
        $crate::value::owned::Value::from($other)
    };
}

// Rocket relies on this because they export their own `json!` with a different
// doc comment than ours, and various Rust bugs prevent them from calling our
// `json!` from their `json!` so they call `json_internal!` directly. Check with
// @SergioBenitez before making breaking changes to this macro.
//
// Changes are fine as long as `json_internal!` does not call any new helper
// macros and can still be invoked as `json_internal!($($json)+)`.
#[cfg(feature = "serde_impl")]
#[macro_export(local_inner_macros)]
#[doc(hidden)]
macro_rules! json_internal_borrowed {
    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an array [...]. Produces a vec![...]
    // of the elements.
    //
    // Must be invoked as: json_internal_borrowed!(@array [] $($tt)*)
    //////////////////////////////////////////////////////////////////////////

    // Done with trailing comma.
    (@array [$($elems:expr_2021,)*]) => {
        json_internal_vec![$($elems,)*]
    };

    // Done without trailing comma.
    (@array [$($elems:expr_2021),*]) => {
        json_internal_vec![$($elems),*]
    };

    // Next element is `null`.
    (@array [$($elems:expr_2021,)*] null $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!(null)] $($rest)*)
    };

    // Next element is `true`.
    (@array [$($elems:expr_2021,)*] true $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!(true)] $($rest)*)
    };

    // Next element is `false`.
    (@array [$($elems:expr_2021,)*] false $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!(false)] $($rest)*)
    };

    // Next element is an array.
    (@array [$($elems:expr_2021,)*] [$($array:tt)*] $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!([$($array)*])] $($rest)*)
    };

    // Next element is a map.
    (@array [$($elems:expr_2021,)*] {$($map:tt)*} $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!({$($map)*})] $($rest)*)
    };

    // Next element is an expression followed by comma.
    (@array [$($elems:expr_2021,)*] $next:expr_2021, $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!($next),] $($rest)*)
    };

    // Last element is an expression with no trailing comma.
    (@array [$($elems:expr_2021,)*] $last:expr_2021) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!($last)])
    };

    // Comma after the most recent element.
    (@array [$($elems:expr_2021),*] , $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)*] $($rest)*)
    };

    // Unexpected token after most recent element.
    (@array [$($elems:expr_2021),*] $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected)
    };

    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an object {...}. Each entry is
    // inserted into the given map variable. Entries are parsed from the TT
    // and then inserted into a stack of entries to be appended all at once
    // at the end, allowing us to pre-allocate the map.
    //
    // Must be invoked as: json_internal!(@object $map () ($($tt)*) ($($tt)*))
    //
    // We require two copies of the input tokens so that we can match on one
    // copy and trigger errors on the other copy.
    //////////////////////////////////////////////////////////////////////////

    // Done.
    (@object $object:ident [@entries] () () ()) => {};

    // Count the number of entries to insert.
    // Modified version of https://docs.rs/halfbrown/0.1.11/src/halfbrown/macros.rs.html#46-62
    (@object @count [@entries $(($value:expr_2021 => $($key:tt)+))*]) => {
        <[()]>::len(&[$(json_internal_borrowed!(@object @count @single ($($key)+))),*])
    };
    (@object @count @single ($($key:tt)+)) => {()};

    // Done. Insert all entries from the stack
    (@object $object:ident [@entries $(($value:expr_2021 => $($key:tt)+))*] () () ()) => {
        let len = json_internal_borrowed!(@object @count [@entries $(($value => $($key)+))*]);

        $object = $crate::value::borrowed::Object::with_capacity_and_hasher(len, $crate::value::ObjectHasher::default());
        $(
            #[allow(let_underscore_drop)]
            let _:Option<_> = $object.insert(($($key)+).into(), $value);
        )*

    };

    // Insert the current entry (followed by trailing comma) into the stack.
    // Entries are inserted in reverse order, the captured $entries is expanded first,
    // keeping inserts in the same order that they're defined in the macro.
    //
    // We expand the $key _after_ the $value since $key => $value leads to a parsing ambiguity.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr_2021) , $($rest:tt)*) => {
        json_internal_borrowed!(@object $object [@entries $($entries)* ($value => $($key)+) ] () ($($rest)*) ($($rest)*));
    };

    // Insert the last entry (without trailing comma) into the stack.
    // At this point the recursion is complete and the entries can now be inserted.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr_2021)) => {
        json_internal_borrowed!(@object $object [@entries $($entries)* ($value => $($key)+) ] () () ());
    };

    // Current entry followed by unexpected token.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr_2021) $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected);
    };


    // Next value is `null`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!(null)) $($rest)*);
    };

    // Next value is `true`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!(true)) $($rest)*);
    };

    // Next value is `false`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!(false)) $($rest)*);
    };

    // Next value is an array.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!([$($array)*])) $($rest)*);
    };

    // Next value is a map.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!({$($map)*})) $($rest)*);
    };

    // Next value is an expression followed by comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr_2021 , $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!($value)) , $($rest)*);
    };

    // Last value is an expression with no trailing comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr_2021) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!($value)));
    };

    // Missing value for last entry. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (:) $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_borrowed!();
    };

    // Missing colon and value for last entry. Trigger a reasonable error
    // message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) () $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_borrowed!();
    };

    // Misplaced colon. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] () (: $($rest:tt)*) ($colon:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `:`".
        json_unexpected!($colon);
    };

    // Found a comma inside a key. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) (, $($rest:tt)*) ($comma:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `,`".
        json_unexpected!($comma);
    };

    // Key is fully parenthesized. This avoids clippy double_parens false
    // positives because the parenthesization may be necessary here.
    (@object $object:ident [@entries $($entries:tt)*] () (($key:expr_2021) : $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] ($key) (: $($rest)*) (: $($rest)*));
    };

    // Munch a token into the current key.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    //////////////////////////////////////////////////////////////////////////
    // The main implementation.
    //
    // Must be invoked as: json_internal!($($json)+)
    //////////////////////////////////////////////////////////////////////////

    (null) => {
        $crate::value::borrowed::Value::Static($crate::StaticNode::Null)
    };

    (true) => {
        $crate::value::borrowed::Value::Static($crate::StaticNode::Bool(true))
    };

    (false) => {
        $crate::value::borrowed::Value::Static($crate::StaticNode::Bool(false))
    };

    ([]) => {
        $crate::value::borrowed::Value::Array(Box::new(json_internal_vec![]))
    };

    ([ $($tt:tt)+ ]) => {
        $crate::value::borrowed::Value::Array(Box::new(json_internal_borrowed!(@array [] $($tt)+)))
    };

    ({}) => {
        {
            use $crate::value::ValueBuilder;
            $crate::value::borrowed::Value::object()
        }
    };

    ({ $($tt:tt)+ }) => {
        $crate::value::borrowed::Value::from({
            let mut object;
            json_internal_borrowed!(@object object [@entries] () ($($tt)+) ($($tt)+));
            object
        })
    };

    // Any Serialize type: numbers, strings, struct literals, variables etc.
    // Must be below every other rule.
    ($other:expr_2021) => {
        $crate::serde::to_borrowed_value(&$other).expect("serde::to_borrowed_value")
    };
}

#[cfg(not(feature = "serde_impl"))]
#[macro_export(local_inner_macros)]
#[doc(hidden)]
macro_rules! json_internal_borrowed {
    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an array [...]. Produces a vec![...]
    // of the elements.
    //
    // Must be invoked as: json_internal_borrowed!(@array [] $($tt)*)
    //////////////////////////////////////////////////////////////////////////

    // Done with trailing comma.
    (@array [$($elems:expr,)*]) => {
        json_internal_vec![$($elems,)*]
    };

    // Done without trailing comma.
    (@array [$($elems:expr),*]) => {
        json_internal_vec![$($elems),*]
    };

    // Next element is `null`.
    (@array [$($elems:expr,)*] null $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!(null)] $($rest)*)
    };

    // Next element is `true`.
    (@array [$($elems:expr,)*] true $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!(true)] $($rest)*)
    };

    // Next element is `false`.
    (@array [$($elems:expr,)*] false $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!(false)] $($rest)*)
    };

    // Next element is an array.
    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!([$($array)*])] $($rest)*)
    };

    // Next element is a map.
    (@array [$($elems:expr,)*] {$($map:tt)*} $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!({$($map)*})] $($rest)*)
    };

    // Next element is an expression followed by comma.
    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!($next),] $($rest)*)
    };

    // Last element is an expression with no trailing comma.
    (@array [$($elems:expr,)*] $last:expr) => {
        json_internal_borrowed!(@array [$($elems,)* json_internal_borrowed!($last)])
    };

    // Comma after the most recent element.
    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        json_internal_borrowed!(@array [$($elems,)*] $($rest)*)
    };

    // Unexpected token after most recent element.
    (@array [$($elems:expr),*] $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected)
    };

    //////////////////////////////////////////////////////////////////////////
    // TT muncher for parsing the inside of an object {...}. Each entry is
    // inserted into the given map variable. Entries are parsed from the TT
    // and then inserted into a stack of entries to be appended all at once
    // at the end, allowing us to pre-allocate the map.
    //
    // Must be invoked as: json_internal!(@object $map () ($($tt)*) ($($tt)*))
    //
    // We require two copies of the input tokens so that we can match on one
    // copy and trigger errors on the other copy.
    //////////////////////////////////////////////////////////////////////////

    // Done.
    (@object $object:ident [@entries] () () ()) => {};

    // Count the number of entries to insert.
    // Modified version of https://docs.rs/halfbrown/0.1.11/src/halfbrown/macros.rs.html#46-62
    (@object @count [@entries $(($value:expr => $($key:tt)+))*]) => {
        <[()]>::len(&[$(json_internal_borrowed!(@object @count @single ($($key)+))),*])
    };
    (@object @count @single ($($key:tt)+)) => {()};

    // Done. Insert all entries from the stack
    (@object $object:ident [@entries $(($value:expr => $($key:tt)+))*] () () ()) => {
        let len = json_internal_borrowed!(@object @count [@entries $(($value => $($key)+))*]);

        $object = $crate::value::borrowed::Object::with_capacity_and_hasher(len, $crate::value::ObjectHasher::default());
        $(
            #[allow(clippy::let_underscore_drop)]
            let _:Option<_> = $object.insert(($($key)+).into(), $value);
        )*

    };

    // Insert the current entry (followed by trailing comma) into the stack.
    // Entries are inserted in reverse order, the captured $entries is expanded first,
    // keeping inserts in the same order that they're defined in the macro.
    //
    // We expand the $key _after_ the $value since $key => $value leads to a parsing ambiguity.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        json_internal_borrowed!(@object $object [@entries $($entries)* ($value => $($key)+) ] () ($($rest)*) ($($rest)*));
    };

    // Insert the last entry (without trailing comma) into the stack.
    // At this point the recursion is complete and the entries can now be inserted.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr)) => {
        json_internal_borrowed!(@object $object [@entries $($entries)* ($value => $($key)+) ] () () ());
    };

    // Current entry followed by unexpected token.
    (@object $object:ident [@entries $($entries:tt)*] [$($key:tt)+] ($value:expr) $unexpected:tt $($rest:tt)*) => {
        json_unexpected!($unexpected);
    };


    // Next value is `null`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!(null)) $($rest)*);
    };

    // Next value is `true`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!(true)) $($rest)*);
    };

    // Next value is `false`.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!(false)) $($rest)*);
    };

    // Next value is an array.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!([$($array)*])) $($rest)*);
    };

    // Next value is a map.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!({$($map)*})) $($rest)*);
    };

    // Next value is an expression followed by comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr , $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!($value)) , $($rest)*);
    };

    // Last value is an expression with no trailing comma.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (: $value:expr) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] [$($key)+] (json_internal_borrowed!($value)));
    };

    // Missing value for last entry. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) (:) $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_borrowed!();
    };

    // Missing colon and value for last entry. Trigger a reasonable error
    // message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)+) () $copy:tt) => {
        // "unexpected end of macro invocation"
        json_internal_borrowed!();
    };

    // Misplaced colon. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] () (: $($rest:tt)*) ($colon:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `:`".
        json_unexpected!($colon);
    };

    // Found a comma inside a key. Trigger a reasonable error message.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) (, $($rest:tt)*) ($comma:tt $($copy:tt)*)) => {
        // Takes no arguments so "no rules expected the token `,`".
        json_unexpected!($comma);
    };

    // Key is fully parenthesized. This avoids clippy double_parens false
    // positives because the parenthesization may be necessary here.
    (@object $object:ident [@entries $($entries:tt)*] () (($key:expr) : $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] ($key) (: $($rest)*) (: $($rest)*));
    };

    // Munch a token into the current key.
    (@object $object:ident [@entries $($entries:tt)*] ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        json_internal_borrowed!(@object $object [@entries $($entries)*] ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    //////////////////////////////////////////////////////////////////////////
    // The main implementation.
    //
    // Must be invoked as: json_internal!($($json)+)
    //////////////////////////////////////////////////////////////////////////

    (null) => {
        $crate::value::borrowed::Value::Static($crate::StaticNode::Null)
    };

    (true) => {
        $crate::value::borrowed::Value::Static($crate::StaticNode::Bool(true))
    };

    (false) => {
        $crate::value::borrowed::Value::Static($crate::StaticNode::Bool(false))
    };

    ([]) => {
        $crate::value::borrowed::Value::Array(Box::new(json_internal_vec![]))
    };

    ([ $($tt:tt)+ ]) => {
        $crate::value::borrowed::Value::Array(Box::new(json_internal_borrowed!(@array [] $($tt)+)))
    };

    ({}) => {
        {
            use $crate::value::ValueBuilder;
            $crate::value::borrowed::Value::object()
        }
    };

    ({ $($tt:tt)+ }) => {
        $crate::value::borrowed::Value::from({
            let mut object;
            json_internal_borrowed!(@object object [@entries] () ($($tt)+) ($($tt)+));
            object
        })
    };

    // Any Serialize type: numbers, strings, struct literals, variables etc.
    // Must be below every other rule.
    ($other:expr) => {
        $crate::value::borrowed::Value::from($other)
    };
}

// The json_internal macro above cannot invoke vec directly because it uses
// local_inner_macros. A vec invocation there would resolve to $crate::vec.
// Instead invoke vec here outside of local_inner_macros.
#[macro_export]
#[doc(hidden)]
macro_rules! json_internal_vec {
    ($($content:tt)*) => {
        vec![$($content)*]
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! json_unexpected {
    () => {};
}

/// possible compiler hint that a branch is unlikely
#[cfg(feature = "hints")]
macro_rules! unlikely {
    ($e:expr) => {{ ::std::intrinsics::unlikely($e) }};
}

/// possible compiler hint that a branch is unlikely
///
/// Technique borrowed from here: <https://github.com/rust-lang/hashbrown/pull/209>
#[cfg(not(feature = "hints"))]
macro_rules! unlikely {
    ($e:expr_2021) => {{
        #[inline]
        #[cold]
        fn cold() {}

        let cond = $e;

        if cond {
            cold();
        }

        cond
    }};
}

pub(crate) use unlikely;

/// static cast to an i32
#[allow(unused_macros)]
macro_rules! static_cast_i32 {
    ($v:expr_2021) => {
        ::std::mem::transmute::<u32, i32>($v)
    };
}
#[allow(unused_imports)]
pub(crate) use static_cast_i32;

/// static cast to an u32
#[allow(unused_macros)]
macro_rules! static_cast_u32 {
    ($v:expr_2021) => {
        // #[allow(clippy::missing_transmute_annotations)]
        ::std::mem::transmute::<i32, u32>($v)
    };
}
#[allow(unused_imports)]
pub(crate) use static_cast_u32;

/// static cast to an i64
macro_rules! static_cast_i64 {
    ($v:expr_2021) => {
        ::std::mem::transmute::<u64, i64>($v)
    };
}
pub(crate) use static_cast_i64;

/// static cast to an i64
#[cfg(all(feature = "approx-number-parsing", feature = "i128"))]
macro_rules! static_cast_i128 {
    ($v:expr_2021) => {
        ::std::mem::transmute::<_, i128>($v)
    };
}
#[cfg(all(feature = "approx-number-parsing", feature = "i128"))]
pub(crate) use static_cast_i128;

/// static cast to an u64
macro_rules! static_cast_u64 {
    ($v:expr_2021) => {
        ::std::mem::transmute::<i64, u64>($v)
    };
}
pub(crate) use static_cast_u64;

/// Custom `try!` macro that does no `From` conversions
///
/// FROM serde-json
/// We only use our own error type; no need for From conversions provided by the
/// standard library's try! macro. This reduces lines of LLVM IR by 4%.
macro_rules! stry {
    ($e:expr_2021) => {
        match $e {
            ::std::result::Result::Ok(val) => val,
            ::std::result::Result::Err(err) => return ::std::result::Result::Err(err),
        }
    };
}
#[allow(unused_imports)]
pub(crate) use stry;

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use crate::{BorrowedValue, OwnedValue};

    #[cfg(feature = "serde_impl")]
    fn owned_test_map() -> OwnedValue {
        let mut h = crate::owned::Object::with_hasher(crate::ObjectHasher::default());
        h.insert("test".into(), 1.into());
        OwnedValue::from(h)
    }
    #[cfg(feature = "serde_impl")]
    fn borrowed_test_map() -> BorrowedValue<'static> {
        let mut h = crate::borrowed::Object::with_hasher(crate::ObjectHasher::default());
        h.insert("test".into(), 1.into());
        BorrowedValue::from(h)
    }

    #[test]
    fn array() {
        let v: OwnedValue = json!(vec![1]);
        assert_eq!(OwnedValue::from(vec![1_u64]), v);
        let v: OwnedValue = json!([1]);
        assert_eq!(BorrowedValue::from(vec![1_u64]), v);
        let v: OwnedValue = json!([]);
        assert_eq!(OwnedValue::array(), v);
    }

    #[cfg(feature = "serde_impl")]
    #[test]
    fn obj() {
        use halfbrown::hashmap;
        let o = owned_test_map();
        let v: OwnedValue = json!(hashmap! {"test" => 1});
        assert_eq!(o, v);
        let v: OwnedValue = json!({"test": 1});
        assert_eq!(o, v);
        let v: OwnedValue = json!({});
        assert_eq!(OwnedValue::object(), v);
    }

    #[test]
    fn array_typed_owned() {
        let v: OwnedValue = json_typed!(owned, vec![1]);
        assert_eq!(OwnedValue::from(vec![1_u64]), v);
        let v: OwnedValue = json_typed!(owned, [1]);
        assert_eq!(OwnedValue::from(vec![1_u64]), v);
        let v: OwnedValue = json_typed!(owned, []);
        assert_eq!(OwnedValue::array(), v);
    }

    #[test]
    fn array_typed_borrowed() {
        let v: BorrowedValue = json_typed!(borrowed, vec![1]);
        assert_eq!(BorrowedValue::from(vec![1_u64]), v);
        let v: BorrowedValue = json_typed!(borrowed, [1]);
        assert_eq!(BorrowedValue::from(vec![1_u64]), v);
        let v: BorrowedValue = json_typed!(borrowed, []);
        assert_eq!(BorrowedValue::array(), v);
    }
    #[cfg(feature = "serde_impl")]
    #[test]
    fn obj_typed_owned() {
        use halfbrown::hashmap;
        let v: OwnedValue = json_typed!(owned, hashmap! {"test" => 1});
        let o = owned_test_map();
        assert_eq!(o, v);
        let v: OwnedValue = json_typed!(owned, {"test": 1});
        assert_eq!(o, v);
        let v: OwnedValue = json_typed!(owned, {});
        assert_eq!(OwnedValue::object(), v);
    }

    #[cfg(feature = "serde_impl")]
    #[test]
    fn obj_typed_borrowed() {
        use halfbrown::hashmap;
        let v: BorrowedValue = json_typed!(borrowed, hashmap! {"test" => 1});
        let o = borrowed_test_map();
        assert_eq!(o, v);
        let v: BorrowedValue = json_typed!(borrowed, {"test": 1});
        assert_eq!(o, v);
        let v: BorrowedValue = json_typed!(borrowed, {});
        assert_eq!(BorrowedValue::object(), v);
    }
}
