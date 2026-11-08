use super::Value;
use crate::Node;

#[derive(Clone, Copy)]
/// Wrapper around the tape that allows interacting with it via a `Array`-like API.
pub struct Array<'tape, 'input>(pub(super) &'tape [Node<'input>]);

pub struct Iter<'tape, 'input>(&'tape [Node<'input>]);

impl<'tape, 'input> Iterator for Iter<'tape, 'input> {
    type Item = Value<'tape, 'input>;

    fn next(&mut self) -> Option<Self::Item> {
        let (head, tail) = self.0.split_at(self.0.first()?.count());
        self.0 = tail;
        Some(Value(head))
    }
}

// value_trait::Array for
impl<'tape, 'input> Array<'tape, 'input>
where
    'input: 'tape,
{
    /// Gets a ref to a value based on n index, returns `None` if the
    /// current Value isn't an Array or doesn't contain the index
    /// it was asked for.
    #[must_use]
    pub fn get(&self, mut idx: usize) -> Option<Value<'tape, 'input>> {
        let mut offset = 1;
        while idx > 0 {
            offset += self.0.get(offset)?.count();
            idx -= 1;
        }
        let count = self.0.get(offset)?.count();
        Some(Value(&self.0[offset..offset + count]))
    }

    /// Iterates over the values paris
    #[must_use]
    pub fn iter<'i>(&'i self) -> Iter<'tape, 'input> {
        Iter(&self.0[1..])
    }

    /// Number of key/value pairs
    /// # Panics
    /// if the tape is not an array
    #[must_use]
    pub fn len(&self) -> usize {
        if let Some(Node::Array { len, .. }) = self.0.first() {
            *len
        } else {
            panic!("invalid tape array")
        }
    }
    /// Returns if the array is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'tape, 'input> IntoIterator for &Array<'tape, 'input> {
    type IntoIter = Iter<'tape, 'input>;
    type Item = Value<'tape, 'input>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
