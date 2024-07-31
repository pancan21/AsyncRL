use std::ops::{Index, IndexMut};

use smallvec::{SmallVec, ToSmallVec};

/// The size of the [`SmallVec`] to use in [`Rope`] and [`RopeMut`].
const SMALLVEC_LEN: usize = 4;

/// A "rope" of immutable slices.
#[derive(Debug, Clone)]
pub struct Rope<'a, S> {
    /// Offsets into the slices.
    offsets: SmallVec<[usize; SMALLVEC_LEN]>,
    /// The immutable slices in question.
    data: SmallVec<[&'a [S]; SMALLVEC_LEN]>,
}

/// A "rope" of mutable slices.
#[derive(Debug)]
pub struct RopeMut<'a, S> {
    /// Offsets into the slices.
    offsets: SmallVec<[usize; SMALLVEC_LEN]>,
    /// The mutable slices in question.
    data: SmallVec<[&'a mut [S]; SMALLVEC_LEN]>,
}

impl<'a, S> Rope<'a, S> {
    /// Create a new [`Rope`] containing data from a vector of immutable slices.
    pub fn new(data: &[&'a [S]]) -> Self {
        let data: SmallVec<[&[S]; SMALLVEC_LEN]> = data.to_smallvec();
        Self {
            offsets: data
                .iter()
                .map(|i| i.len())
                .scan(0, |a, b| {
                    let ret = Some(*a);
                    *a += b;
                    ret
                })
                .collect(),
            data,
        }
    }

    /// Get the length of the [`Rope`].
    pub fn len(&self) -> usize {
        self.offsets
            .last()
            .zip(self.data.last())
            .map(|(i, data)| i + data.len())
            .unwrap_or(0)
    }

    /// Checks if the [`Rope`] is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Merge two [`Rope`] together.
    pub fn merge(mut self, rope2: Rope<'a, S>) -> Rope<'a, S> {
        let len = self.len();
        self.offsets
            .extend(rope2.offsets.into_iter().map(|i| i + len));
        self.data.extend(rope2.data);

        self
    }
}

impl<'a, S> RopeMut<'a, S> {
    /// Create a new [`RopeMut`] containing data from a vector of mutable slices.
    pub fn new<const N: usize>(data: [&'a mut [S]; N]) -> Self {
        Self {
            offsets: data
                .iter()
                .map(|i| i.len())
                .scan(0, |a, b| {
                    let ret = Some(*a);
                    *a += b;
                    ret
                })
                .collect(),
            data: data.into_iter().collect(),
        }
    }

    /// Get the length of the [`RopeMut`].
    pub fn len(&self) -> usize {
        self.offsets
            .last()
            .zip(self.data.last())
            .map(|(i, data)| i + data.len())
            .unwrap_or(0)
    }

    /// Checks if the [`RopeMut`] is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Merge two [`RopeMut`] together.
    pub fn merge(mut self, rope2: RopeMut<'a, S>) -> RopeMut<'a, S> {
        let len = self.len();
        self.offsets
            .extend(rope2.offsets.into_iter().map(|i| i + len));
        self.data.extend(rope2.data);

        self
    }

    /// Clones the data from the slice into the underlying data in the [`RopeMut`]
    ///
    /// # Panics
    /// If the slice doesn't have the same length as the [`RopeMut`].
    pub fn clone_from_slice(&mut self, slice: &[S])
    where
        S: Clone,
    {
        assert_eq!(
            self.len(),
            slice.len(),
            "Expected `self` and `slice` to have the same length but got {} and {}, respectively",
            self.len(),
            slice.len()
        );

        self.offsets
            .iter()
            .zip(self.data.iter_mut())
            .for_each(|(&offset, data)| {
                let len = data.len();
                data.clone_from_slice(&slice[offset..(offset + len)])
            })
    }

    /// Copies the data from the slice into the underlying data in the [`RopeMut`]
    ///
    /// # Panics
    /// If the slice doesn't have the same length as the [`RopeMut`].
    pub fn copy_from_slice(&mut self, slice: &[S])
    where
        S: Copy,
    {
        assert_eq!(
            self.len(),
            slice.len(),
            "Expected `self` and `slice` to have the same length but got {} and {}, respectively",
            self.len(),
            slice.len()
        );

        self.offsets
            .iter()
            .zip(self.data.iter_mut())
            .for_each(|(&offset, data)| {
                let len = data.len();
                data.copy_from_slice(&slice[offset..(offset + len)])
            })
    }
}

impl<'a, S> From<RopeMut<'a, S>> for Rope<'a, S> {
    fn from(value: RopeMut<'a, S>) -> Self {
        Self {
            offsets: value.offsets,
            data: value.data.into_iter().map(|i| &*i).collect(),
        }
    }
}

impl<'a, S> Index<usize> for Rope<'a, S> {
    type Output = S;

    fn index(&self, index: usize) -> &Self::Output {
        let idx = self.offsets.partition_point(|&i| i < index) - 1;
        &self.data[idx][index - self.offsets[idx]]
    }
}

impl<'a, S> Index<usize> for RopeMut<'a, S> {
    type Output = S;

    fn index(&self, index: usize) -> &Self::Output {
        let idx = self.offsets.partition_point(|&i| i < index) - 1;
        &self.data[idx][index - self.offsets[idx]]
    }
}

impl<'a, S> IndexMut<usize> for RopeMut<'a, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = self.offsets.partition_point(|&i| i < index) - 1;
        &mut self.data[idx][index - self.offsets[idx]]
    }
}

/// Implements iterator for [`Rope<S>`].
pub struct RopeIterator<'a, S> {
    /// Internal [`Rope`] Instance
    rope: Rope<'a, S>,
    /// Current Slice
    current: Option<&'a [S]>,
}

/// Implements iterator for [`RopeMut<S>`]
pub struct RopeIteratorMut<'a, S> {
    /// Internal [`RopeMut`] Instance
    rope: RopeMut<'a, S>,
    /// Current Slice
    current: Option<&'a mut [S]>,
}

impl<'a, S> IntoIterator for Rope<'a, S> {
    type Item = &'a S;

    type IntoIter = RopeIterator<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        RopeIterator {
            rope: self,
            current: None,
        }
    }
}

impl<'a, S> IntoIterator for RopeMut<'a, S> {
    type Item = &'a mut S;

    type IntoIter = RopeIteratorMut<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        RopeIteratorMut {
            rope: self,
            current: None,
        }
    }
}

impl<'a, S> Iterator for RopeIterator<'a, S> {
    type Item = &'a S;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
            Some(&[]) | None => {
                if let Some(data) = self.rope.data.pop() {
                    self.current.replace(&data[1..]);
                    Some(&data[0])
                } else {
                    None
                }
            }
            Some(data) => {
                self.current.replace(&data[1..]);
                Some(&data[0])
            }
        }
    }
}

impl<'a, S> Iterator for RopeIteratorMut<'a, S> {
    type Item = &'a mut S;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current.take() {
            Some(&mut []) | None => {
                if let Some(data) = self.rope.data.pop() {
                    let (a, b) = data.split_at_mut(1);
                    self.current.replace(b);
                    Some(&mut a[0])
                } else {
                    None
                }
            }
            Some(data) => {
                let (a, b) = data.split_at_mut(1);
                self.current.replace(b);
                Some(&mut a[0])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rope;

    #[test]
    fn test_rope_simple() {
        let data = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let rope = Rope::new(&[&data]);

        todo!();
    }
}
