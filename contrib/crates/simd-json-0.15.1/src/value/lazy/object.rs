use std::{
    borrow::{Borrow, Cow},
    hash::Hash,
};

use super::Value;
use crate::{borrowed, tape};

/// Wrapper around the tape that allows interacting with it via a `Object`-like API.
pub enum Object<'borrow, 'tape, 'input> {
    /// Tape variant
    Tape(tape::Object<'tape, 'input>),
    /// Value variant
    Value(&'borrow borrowed::Object<'input>),
}
/// Iterator over key valye paris in an object
pub enum Iter<'borrow, 'tape, 'input> {
    /// Tape variant
    Tape(tape::object::Iter<'tape, 'input>),
    /// Value variant
    Value(halfbrown::Iter<'borrow, crate::cow::Cow<'input, str>, borrowed::Value<'input>>),
}

/// Iterator over the keys of an object
pub enum Keys<'borrow, 'tape, 'input> {
    /// Tape variant
    Tape(tape::object::Keys<'tape, 'input>),
    /// Value variant
    Value(halfbrown::Keys<'borrow, crate::cow::Cow<'input, str>, borrowed::Value<'input>>),
}
/// Iterator over the values of an object
pub enum Values<'borrow, 'tape, 'input> {
    /// Tape variant
    Tape(tape::object::Values<'tape, 'input>),
    /// Value variant
    Value(halfbrown::Values<'borrow, crate::cow::Cow<'input, str>, borrowed::Value<'input>>),
}

//value_trait::Object for
impl<'tape, 'input> Object<'_, 'tape, 'input> {
    /// Gets a ref to a value based on a key, returns `None` if the
    /// current Value isn't an Object or doesn't contain the key
    /// it was asked for.
    #[must_use]
    pub fn get<'a, Q>(&'a self, k: &Q) -> Option<Value<'a, 'tape, 'input>>
    where
        str: Borrow<Q>,
        for<'b> crate::cow::Cow<'b, str>: Borrow<Q>,
        Q: ?Sized + Hash + Eq + Ord,
    {
        match self {
            Object::Tape(t) => t.get(k).map(Value::Tape),
            Object::Value(v) => v.get(k).map(Cow::Borrowed).map(Value::Value),
        }
    }

    /// Iterates over the key value paris
    #[allow(clippy::pedantic)] // we want into_iter_without_iter but that lint doesn't exist in older clippy
    #[must_use]
    pub fn iter<'i>(&'i self) -> Iter<'i, 'tape, 'input> {
        match self {
            Object::Tape(t) => Iter::Tape(t.iter()),
            Object::Value(v) => Iter::Value(v.iter()),
        }
    }

    /// Iterates over the keys
    #[must_use]
    pub fn keys<'i>(&'i self) -> Keys<'i, 'tape, 'input> {
        match self {
            Object::Tape(t) => Keys::Tape(t.keys()),
            Object::Value(v) => Keys::Value(v.keys()),
        }
    }

    /// Iterates over the values
    #[must_use]
    pub fn values<'i>(&'i self) -> Values<'i, 'tape, 'input> {
        match self {
            Object::Tape(t) => Values::Tape(t.values()),
            Object::Value(v) => Values::Value(v.values()),
        }
    }

    /// Number of key/value pairs
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Object::Tape(t) => t.len(),
            Object::Value(v) => v.len(),
        }
    }

    /// Returns if the object is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// impl<'tape, 'input> IntoIterator for &Object<'tape, 'input> {
//     type IntoIter = Iter<'tape, 'input>;
//     type Item = (&'input str, Value<'tape, 'input>);
//     fn into_iter(self) -> Self::IntoIter {
//         self.iter()
//     }
// }

impl<'borrow, 'tape, 'input> Iterator for Iter<'borrow, 'tape, 'input> {
    type Item = (&'borrow str, Value<'borrow, 'tape, 'input>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Tape(t) => t.next().map(|(k, v)| (k, Value::Tape(v))),
            Iter::Value(v) => v
                .next()
                .map(|(k, v)| (k.as_ref(), Value::Value(Cow::Borrowed(v)))),
        }
    }
}

impl<'borrow> Iterator for Keys<'borrow, '_, '_> {
    type Item = &'borrow str;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Keys::Tape(t) => t.next(),
            Keys::Value(v) => v.next().map(std::convert::AsRef::as_ref),
        }
    }
}

impl<'borrow, 'tape, 'input> Iterator for Values<'borrow, 'tape, 'input> {
    type Item = Value<'borrow, 'tape, 'input>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Values::Tape(t) => t.next().map(Value::Tape),
            Values::Value(v) => v.next().map(|v| Value::Value(Cow::Borrowed(v))),
        }
    }
}

#[cfg(test)]
mod test {
    use value_trait::base::ValueAsScalar;

    use crate::to_tape;

    #[test]
    fn get_ints() -> crate::Result<()> {
        let mut input = br#"{"snot": 1, "badger":2, "cake":3, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let a = v.as_object().expect("is an object");
        assert_eq!(a.get("snot").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(a.get("badger").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(a.get("cake").and_then(|v| v.as_u64()), Some(3));
        assert_eq!(a.get("cookie").and_then(|v| v.as_u64()), Some(4));
        assert_eq!(a.get("monster"), None);
        Ok(())
    }

    #[test]
    fn get_container() -> crate::Result<()> {
        let mut input =
            br#"{"snot": 1, "badger":[2, 2.5], "cake":{"frosting": 3}, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let a = v.as_object().expect("is an object");
        assert_eq!(a.get("snot").and_then(|v| v.as_u64()), Some(1));
        let badger = a.get("badger").expect("is an array");
        let badger = badger.as_array().expect("is an array");
        assert_eq!(badger.get(0).and_then(|v| v.as_u64()), Some(2));
        assert_eq!(badger.get(1).and_then(|v| v.as_f64()), Some(2.5));
        let cake = a.get("cake").expect("is an object");
        let cake = cake.as_object().expect("is an object");
        assert_eq!(cake.get("frosting").and_then(|v| v.as_u64()), Some(3));
        assert_eq!(a.get("cookie").and_then(|v| v.as_u64()), Some(4));
        assert_eq!(a.get("monster"), None);
        Ok(())
    }
    #[test]
    fn iter_ints() -> crate::Result<()> {
        let mut input = br#"{"snot": 1, "badger":2, "cake":3, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let v = v
            .as_object()
            .expect("is an object")
            .iter()
            .map(|(k, v)| (k, v.as_u64().expect("integer")))
            .collect::<Vec<_>>();
        assert_eq!(
            v,
            vec![("snot", 1), ("badger", 2), ("cake", 3), ("cookie", 4)]
        );

        Ok(())
    }

    #[test]
    fn keys_ints() -> crate::Result<()> {
        let mut input = br#"{"snot": 1, "badger":2, "cake":3, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let v = v
            .as_object()
            .expect("is an object")
            .keys()
            .collect::<Vec<_>>();
        assert_eq!(v, vec!["snot", "badger", "cake", "cookie"]);

        Ok(())
    }

    #[test]
    fn values_ints() -> crate::Result<()> {
        let mut input = br#"{"snot": 1, "badger":2, "cake":3, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let v = v
            .as_object()
            .expect("is an object")
            .values()
            .map(|v| v.as_u64().expect("integer"))
            .collect::<Vec<_>>();
        assert_eq!(v, vec![1, 2, 3, 4]);

        Ok(())
    }
    #[test]
    fn iter_container() -> crate::Result<()> {
        let mut input =
            br#"{"snot": 1, "badger":[2, 2.5], "cake":{"frosting": 3}, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let v = v
            .as_object()
            .expect("is an object")
            .iter()
            .map(|(k, v)| (k, v.as_u64()))
            .collect::<Vec<_>>();
        assert_eq!(
            v,
            vec![
                ("snot", Some(1)),
                ("badger", None),
                ("cake", None),
                ("cookie", Some(4))
            ]
        );
        Ok(())
    }
    #[test]
    fn keys_container() -> crate::Result<()> {
        let mut input =
            br#"{"snot": 1, "badger":[2, 2.5], "cake":{"frosting": 3}, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let v = v
            .as_object()
            .expect("is an object")
            .keys()
            .collect::<Vec<_>>();
        assert_eq!(v, vec!["snot", "badger", "cake", "cookie"]);

        Ok(())
    }

    #[test]
    fn values_container() -> crate::Result<()> {
        let mut input =
            br#"{"snot": 1, "badger":[2, 2.5], "cake":{"frosting": 3}, "cookie":4}"#.to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let v = v
            .as_object()
            .expect("is an object")
            .values()
            .map(|v| v.as_u64())
            .collect::<Vec<_>>();
        assert_eq!(v, vec![Some(1), None, None, Some(4)]);

        Ok(())
    }
}
