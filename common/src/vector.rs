use bytemuck::{Pod, Zeroable};

use std::ops::{Index, IndexMut};

use crate::{
    rope::{Rope, RopeMut},
    system::DynamicVector,
    Float,
};

/// This attempts to compile invalid types
/// ```compile_fail
/// use rtdriver::Vector;
/// fn test_addition_invalid_types() {
///     let x = Vector([0f64, 1.5f64]);
///     let y = Vector([1.3f32, 22.5f32]);
///     assert_eq!(x + y, Vector([1.3, 24.0]));
/// }
/// ```
///
#[derive(Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(transparent)]
pub struct Vector<T, const DIMS: usize>([T; DIMS]);

impl<T, const DIMS: usize> Clone for Vector<T, DIMS>
where
    [T; DIMS]: Clone,
{
    fn clone(&self) -> Self {
        Vector(self.0.clone())
    }
}
impl<T, const DIMS: usize> Copy for Vector<T, DIMS> where [T; DIMS]: Copy {}

impl<T: Default, const DIMS: usize> Default for Vector<T, DIMS> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| T::default()))
    }
}

impl<T: num::Zero, const DIMS: usize> Vector<T, DIMS> {
    /// Returns a vector with all components zero.
    #[inline]
    pub fn zero() -> Self {
        Vector::from_idx(|_| T::zero())
    }
}

impl<T, const DIMS: usize> Vector<T, DIMS> {
    /// Given a raw array, construct a [`Vector`] wrapping it.
    pub fn new(data: [T; DIMS]) -> Vector<T, DIMS> {
        Self(data)
    }
}

impl<const DIMS: usize> Vector<bool, DIMS> {
    /// For a boolean [`Vector`] ([`Vector<bool, _>`]), check that all elements are true.
    pub fn all(self) -> bool {
        self.0.into_iter().all(std::convert::identity)
    }

    /// For a boolean [`Vector`] ([`Vector<bool, _>`]), check that at least one element is true.
    pub fn any(self) -> bool {
        self.0.into_iter().any(std::convert::identity)
    }
}

impl<T, const DIMS: usize> Vector<T, DIMS> {
    /// Given a scalar `T`, construct a vector [`Vector<T>`] where all elements are the same,
    /// with the value of the scalar.
    pub fn broadcast(value: T) -> Self
    where
        T: Clone,
    {
        Self::from_idx(|_| value.clone())
    }

    /// Gets an interior immutable array
    pub fn as_array(&self) -> &[T; DIMS] {
        &self.0
    }

    /// Gets an interior mutable array
    pub fn as_array_mut(&mut self) -> &mut [T; DIMS] {
        &mut self.0
    }

    /// Produces an iterator given by the underlying slice iterator.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.0.iter()
    }

    /// Given an vector of type `T` and map of type [`Fn(T) -> U`], produces a vector of type
    /// `U` by repeatedly applying the map on each element.
    #[inline]
    pub fn map<U>(self, map_fn: impl Fn(T) -> U) -> Vector<U, DIMS> {
        Vector(self.0.map(map_fn))
    }

    /// Given an vector of type `T` and map of type [`Fn(usize, T) -> T`], produces a vector of
    /// type `T` by repeatedly applying the map on each element's index and value.
    #[inline]
    pub fn update(self, update_fn: impl Fn(usize, T) -> T) -> Self
    where
        T: Copy,
    {
        Self(std::array::from_fn(|i| update_fn(i, self[i])))
    }

    /// Given a map of type [`Fn(usize) -> T`], produces a vector by passing in each index from
    /// `0..DIMS` to the map.
    #[inline]
    pub fn from_idx(idx_fn: impl Fn(usize) -> T) -> Self {
        Self(std::array::from_fn(idx_fn))
    }

    /// Given a type that is "additively reducible", compute the sum over all elements of the
    /// vector of that type. The addition need not be commutative and is performed from left to
    /// right. If the vector is empty, then the additive-identity element of the type is returned.
    #[inline]
    pub fn sum(self) -> T
    where
        T: std::ops::Add<Output = T> + num::Zero,
    {
        self.0.into_iter().reduce(|a, b| a + b).unwrap_or(T::zero())
    }

    /// Given a type that is "multiplicatively reducible", compute the product over all elements of
    /// the vector of that type. The multiplication need not be commutative and is performed from
    /// left to right. If the vector is empty, then the multiplicative-identity element of the type
    /// is returned.
    #[inline]
    pub fn prod(self) -> T
    where
        T: std::ops::Mul<Output = T> + num::One,
    {
        self.0.into_iter().reduce(|a, b| a * b).unwrap_or(T::one())
    }
}

impl<T, const DIMS: usize> IntoIterator for Vector<T, DIMS> {
    fn into_iter(self) -> std::array::IntoIter<T, DIMS> {
        self.0.into_iter()
    }

    type Item = <[T; DIMS] as IntoIterator>::Item;

    type IntoIter = <[T; DIMS] as IntoIterator>::IntoIter;
}

impl<T, const DIMS: usize> Index<usize> for Vector<T, DIMS> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const DIMS: usize> IndexMut<usize> for Vector<T, DIMS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const DIMS: usize> From<[T; DIMS]> for Vector<T, DIMS> {
    fn from(value: [T; DIMS]) -> Self {
        Self(value)
    }
}

impl<T: num::Zero + num::One, const DIMS: usize> Vector<T, DIMS> {
    /// Produces a basis vector in the direction of the `idx`-th dimension.
    pub fn basis(idx: usize) -> Self {
        let mut out = Self::zero();
        out[idx].set_one();

        out
    }
}

/// Given a scalar unary operation, construct the associated vector operation.
macro_rules! impl_unary_operation {
    ($op:ident) => {
        paste::paste! {
            impl<T: Copy, U, const DIMS: usize> std::ops::$op for Vector<T, DIMS>
            where
                T: std::ops::$op<Output = U>,
            {
                type Output = Vector<U, DIMS>;

                fn [< $op:lower >](self) -> Self::Output {
                    Vector::from_idx(|i| self[i].[< $op:lower >]())
                }
            }
        }
    };
}

/// Given a scalar binary operation, construct the associated vector operation for the pairs
/// `(Vector, Vector)` and `(Vector, Scalar)`.
macro_rules! impl_binary_operation {
    ($($op:ident),+$(,)?) => {
        paste::paste! {
            $(impl<T: Copy, U: Copy, V, const DIMS: usize> std::ops::$op<Vector<U, DIMS>> for Vector<T, DIMS>
            where
                T: std::ops::$op<U, Output = V>,
            {
                type Output = Vector<V, DIMS>;

                fn [< $op:lower >](self, rhs: Vector<U, DIMS>) -> Self::Output {
                    Vector::from_idx(|i| self[i].[< $op:lower >](rhs[i]))
                }
            })+
        }

        paste::paste! {
            $(impl<T: Copy, U: Copy + num::Num, V, const DIMS: usize> std::ops::$op<U> for Vector<T, DIMS>
            where
                T: std::ops::$op<U, Output = V>,
            {
                type Output = Vector<V, DIMS>;

                fn [< $op:lower >](self, rhs: U) -> Self::Output {
                    let rhs = Vector::<U, DIMS>::broadcast(rhs);
                    Vector::from_idx(|i| self[i].[< $op:lower >](rhs[i]))
                }
            })+
        }
    };
}

/// Given a scalar binary-assign operation, construct the associated vector operation for the pairs
/// `(Vector, Vector)` and `(Vector, Scalar)`.
macro_rules! impl_binary_assign_operation {
    ($($op:ident),+$(,)?) => {
        paste::paste! {
            $(impl<T: Copy, U: Copy, const DIMS: usize> std::ops::[< $op Assign >]<Vector<U, DIMS>> for Vector<T, DIMS>
            where
                T: std::ops::[< $op >]<U, Output=T>,
            {
                fn [< $op:lower _assign >](&mut self, rhs: Vector<U, DIMS>) {
                    use std::ops::$op;

                    *self = Vector::<T, DIMS>::[< $op:lower >](*self, rhs);
                }
            })+
        }

        paste::paste! {
            $(impl<T: Copy, U: Copy + num::Num, const DIMS: usize> std::ops::[< $op Assign >]<U> for Vector<T, DIMS>
            where
                T: std::ops::[< $op >]<U, Output=T>,
            {
                fn [< $op:lower _assign >](&mut self, rhs: U) {
                    use std::ops::$op;

                    let rhs = Vector::<U, DIMS>::broadcast(rhs);
                    *self = Vector::<T, DIMS>::[< $op:lower >](*self, rhs);
                }
            })+
        }
    };
}

impl_unary_operation!(Neg);
impl_binary_operation!(Add, Sub, Mul, Div, Rem);
impl_binary_assign_operation!(Add, Sub, Mul, Div, Rem);

impl<S: Float, const DIMS: usize> DynamicVector<S> for Vector<S, DIMS> {
    fn copy_from_slice(&mut self, v: &[S]) {
        <[S]>::copy_from_slice(&mut *self.as_array_mut(), v);
    }

    fn get_rope(&self) -> crate::rope::Rope<S> {
        Rope::new(&[self.as_array()])
    }

    fn get_rope_mut(&mut self) -> crate::rope::RopeMut<S> {
        RopeMut::new([self.as_array_mut()])
    }
}

#[cfg(test)]
mod tests {
    use super::Vector;

    #[test]
    fn test_addition_u8() {
        let x = Vector([0u8, 1]);
        let y = Vector([5, 254]);

        assert_eq!(x + y, Vector([5, 255]));
    }

    #[test]
    fn test_addition_u16() {
        let x = Vector([0u16, 1]);
        let y = Vector([5, 254]);

        assert_eq!(x + y, Vector([5, 255]));
    }

    #[test]
    fn test_addition_u32() {
        let x = Vector([0u32, 1]);
        let y = Vector([5, 254]);

        assert_eq!(x + y, Vector([5, 255]));
    }

    #[test]
    fn test_addition_u64() {
        let x = Vector([0u64, 1]);
        let y = Vector([5, 254]);

        assert_eq!(x + y, Vector([5, 255]));
    }

    #[test]
    fn test_addition_u128() {
        let x = Vector([0u128, 1]);
        let y = Vector([5, 254]);

        assert_eq!(x + y, Vector([5, 255]));
    }

    #[test]
    fn test_addition_usize() {
        let x = Vector([0usize, 1]);
        let y = Vector([5, 254]);

        assert_eq!(x + y, Vector([5, 255]));
    }

    #[test]
    fn test_addition_i8() {
        let x = Vector([-5i8, 1]);
        let y = Vector([5, -15]);

        assert_eq!(x + y, Vector([0, -14]));
    }

    #[test]
    fn test_addition_i16() {
        let x = Vector([-5i16, 1]);
        let y = Vector([5, -15]);

        assert_eq!(x + y, Vector([0, -14]));
    }

    #[test]
    fn test_addition_i32() {
        let x = Vector([-5i32, 1]);
        let y = Vector([5, -15]);

        assert_eq!(x + y, Vector([0, -14]));
    }

    #[test]
    fn test_addition_i64() {
        let x = Vector([-5i64, 1]);
        let y = Vector([5, -15]);

        assert_eq!(x + y, Vector([0, -14]));
    }

    #[test]
    fn test_addition_i128() {
        let x = Vector([-5i128, 1]);
        let y = Vector([5, -15]);

        assert_eq!(x + y, Vector([0, -14]));
    }

    #[test]
    fn test_addition_isize() {
        let x = Vector([-5isize, 1]);
        let y = Vector([5, -15]);

        assert_eq!(x + y, Vector([0, -14]));
    }

    #[test]
    fn test_addition_f32() {
        let x = Vector([0f32, 1.5]);
        let y = Vector([1.3, 22.5]);

        assert_eq!(x + y, Vector([1.3, 24.0]));
    }

    #[test]
    fn test_addition_f64() {
        let x = Vector([0f64, 1.5]);
        let y = Vector([1.3, 22.5]);

        assert_eq!(x + y, Vector([1.3, 24.0]));
    }
}
