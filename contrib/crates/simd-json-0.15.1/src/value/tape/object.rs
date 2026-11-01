use std::{borrow::Borrow, hash::Hash};

use super::Value;
use crate::Node;

/// Wrapper around the tape that allows interacting with it via a `Object`-like API.
pub struct Object<'tape, 'input>(pub(super) &'tape [Node<'input>]);

pub struct Iter<'tape, 'input>(&'tape [Node<'input>]);
pub struct Keys<'tape, 'input>(&'tape [Node<'input>]);
pub struct Values<'tape, 'input>(&'tape [Node<'input>]);

//value_trait::Object for
impl<'tape, 'input> Object<'tape, 'input> {
    /// Gets a ref to a value based on a key, returns `None` if the
    /// current Value isn't an Object or doesn't contain the key
    /// it was asked for.
    #[must_use]
    pub fn get<Q>(&self, k: &Q) -> Option<Value<'tape, 'input>>
    where
        str: Borrow<Q> + Hash + Eq,
        Q: ?Sized + Hash + Eq + Ord,
    {
        let &Node::Object { mut len, .. } = self.0.first()? else {
            return None;
        };
        let mut idx = 1;
        while len > 0 {
            let s = self.0.get(idx)?.as_str()?;
            idx += 1;
            len -= 1;
            let count = self.0.get(idx)?.count();
            let s: &Q = s.borrow();
            if s == k {
                return Some(Value(&self.0[idx..idx + count]));
            }
            idx += count;
        }
        None
    }

    /// Iterates over the key value paris
    #[must_use]
    pub fn iter<'i>(&'i self) -> Iter<'tape, 'input> {
        Iter(&self.0[1..])
    }

    /// Iterates over the keys
    #[must_use]
    pub fn keys<'i>(&'i self) -> Keys<'tape, 'input> {
        Keys(&self.0[1..])
    }

    /// Iterates over the values
    #[must_use]
    pub fn values<'i>(&'i self) -> Values<'tape, 'input> {
        Values(&self.0[1..])
    }

    /// Number of key/value pairs
    #[must_use]
    pub fn len(&self) -> usize {
        let Some(Node::Object { len, .. }) = self.0.first() else {
            unreachable!("invalid tape object");
        };
        *len
    }

    /// Returns if the object is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'tape, 'input> IntoIterator for &Object<'tape, 'input> {
    type IntoIter = Iter<'tape, 'input>;
    type Item = (&'input str, Value<'tape, 'input>);
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'tape, 'input> Iterator for Iter<'tape, 'input> {
    type Item = (&'input str, Value<'tape, 'input>);

    fn next(&mut self) -> Option<Self::Item> {
        let (k, v) = self.0.split_first()?;
        let k = k.as_str()?;
        let count = v.first()?.count();
        let (head, tail) = v.split_at(count);
        self.0 = tail;
        Some((k, Value(head)))
    }
}

impl<'input> Iterator for Keys<'_, 'input> {
    type Item = &'input str;
    fn next(&mut self) -> Option<Self::Item> {
        let (k, v) = self.0.split_first()?;
        let k = k.as_str()?;
        let count = v.first()?.count();
        let (_, tail) = v.split_at(count);
        self.0 = tail;
        Some(k)
    }
}

impl<'tape, 'input> Iterator for Values<'tape, 'input> {
    type Item = Value<'tape, 'input>;
    fn next(&mut self) -> Option<Self::Item> {
        let (_, v) = self.0.split_first()?;
        let count = v.first()?.count();
        let (head, tail) = v.split_at(count);
        self.0 = tail;
        Some(Value(head))
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
