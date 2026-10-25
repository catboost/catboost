use std::borrow::Cow;

use super::Value;
use crate::{borrowed, tape};

#[derive(Clone)]
/// Wrapper around the tape that allows interacting with it via a `Array`-like API.
pub enum Array<'borrow, 'tape, 'input> {
    /// Tape variant
    Tape(tape::Array<'tape, 'input>),
    /// Value variant
    Value(&'borrow borrowed::Array<'input>),
}

/// Iterator over the values in an array
pub enum Iter<'borrow, 'tape, 'input> {
    /// Tape variant
    Tape(tape::array::Iter<'tape, 'input>),
    /// Value variant
    Value(std::slice::Iter<'borrow, borrowed::Value<'input>>),
}

impl<'borrow, 'tape, 'input> Iterator for Iter<'borrow, 'tape, 'input> {
    type Item = Value<'borrow, 'tape, 'input>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Tape(t) => t.next().map(Value::Tape),
            Iter::Value(v) => v.next().map(Cow::Borrowed).map(Value::Value),
        }
    }
}

// value_trait::Array for
impl<'tape, 'input> Array<'_, 'tape, 'input> {
    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    pub fn get<'a>(&'a self, idx: usize) -> Option<Value<'a, 'tape, 'input>> {
        match self {
            Array::Tape(t) => t.get(idx).map(Value::Tape),
            Array::Value(v) => v.get(idx).map(Cow::Borrowed).map(Value::Value),
        }
    }
    /// Iterates over the values paris
    #[allow(clippy::pedantic)] // we want into_iter_without_iter but that lint doesn't exist in older clippy
    #[must_use]
    pub fn iter<'i>(&'i self) -> Iter<'i, 'tape, 'input> {
        match self {
            Array::Tape(t) => Iter::Tape(t.iter()),
            Array::Value(v) => Iter::Value(v.iter()),
        }
    }

    /// Number of key/value pairs
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Array::Tape(t) => t.len(),
            Array::Value(v) => v.len(),
        }
    }
    /// Returns if the array is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod test {
    use crate::to_tape;
    use value_trait::base::ValueAsScalar;

    #[test]
    fn get_ints() -> crate::Result<()> {
        let mut input = b"[1,2,3,4]".to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let a = v.as_array().expect("is an array");
        assert_eq!(a.get(0).and_then(|v| v.as_u64()), Some(1));
        assert_eq!(a.get(1).and_then(|v| v.as_u64()), Some(2));
        assert_eq!(a.get(2).and_then(|v| v.as_u64()), Some(3));
        assert_eq!(a.get(3).and_then(|v| v.as_u64()), Some(4));
        assert_eq!(a.get(4), None);
        Ok(())
    }

    #[test]
    fn get_nested() -> crate::Result<()> {
        let mut input = b"[1,[2,3],4]".to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let a = v.as_array().expect("is an array");
        assert_eq!(a.get(0).and_then(|v| v.as_u64()), Some(1));
        let a1 = a.get(1).expect("has first element");
        let a2 = a1.as_array().expect("is an array");
        assert_eq!(a2.get(0).and_then(|v| v.as_u64()), Some(2));
        assert_eq!(a2.get(1).and_then(|v| v.as_u64()), Some(3));
        assert_eq!(a.get(2).and_then(|v| v.as_u64()), Some(4));
        assert_eq!(a.get(3), None);
        Ok(())
    }

    #[test]
    fn iter() -> crate::Result<()> {
        let mut input = b"[1,2,3,4]".to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let a = v.as_array().expect("is an array");
        let v = a
            .iter()
            .map(|v| v.as_u8().expect("integer"))
            .collect::<Vec<_>>();

        assert_eq!(v, vec![1, 2, 3, 4]);

        Ok(())
    }
    #[test]
    fn iter_container() -> crate::Result<()> {
        let mut input = b"[1,[2,3],4]".to_vec();
        let t = to_tape(input.as_mut_slice())?;
        let v = t.as_value();
        let a = v.as_array().expect("is an array");
        let v = a.iter().map(|v| v.as_u8()).collect::<Vec<_>>();

        assert_eq!(v, vec![Some(1), None, Some(4)]);

        Ok(())
    }
}
