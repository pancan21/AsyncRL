use bytemuck::{Pod, Zeroable};

use std::ops::{Index, IndexMut};

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

impl<T: num::Zero, const DIMS: usize> Vector<T, DIMS> {
    #[inline]
    pub fn zero() -> Self {
        Vector::from_idx(|_| T::zero())
    }
}

impl<T, const DIMS: usize> Vector<T, DIMS> {
    pub fn new(data: [T; DIMS]) -> Vector<T, DIMS> {
        Self(data)
    }
}

impl<const DIMS: usize> Vector<bool, DIMS> {
    pub fn all(self) -> bool {
        self.0.into_iter().all(std::convert::identity)
    }
}

impl<T, const DIMS: usize> Vector<T, DIMS> {
    pub fn broadcast(value: T) -> Self
    where
        T: Clone,
    {
        Self::from_idx(|_| value.clone())
    }

    pub fn iter(self) -> impl Iterator<Item = T> {
        self.0.into_iter()
    }

    #[inline]
    pub fn map<U>(self, map_fn: impl Fn(T) -> U) -> Vector<U, DIMS> {
        Vector(self.0.map(map_fn))
    }

    #[inline]
    pub fn update(self, update_fn: impl Fn(usize, T) -> T) -> Self
    where
        T: Copy,
    {
        Self(std::array::from_fn(|i| update_fn(i, self[i])))
    }

    #[inline]
    pub fn from_idx(idx_fn: impl Fn(usize) -> T) -> Self {
        Self(std::array::from_fn(idx_fn))
    }

    #[inline]
    pub fn sum(self) -> T
    where
        T: std::ops::Add<Output = T> + num::Zero,
    {
        self.0.into_iter().reduce(|a, b| a + b).unwrap_or(T::zero())
    }

    #[inline]
    pub fn prod(self) -> T
    where
        T: std::ops::Add<Output = T> + num::Zero,
    {
        self.0.into_iter().reduce(|a, b| a + b).unwrap_or(T::zero())
    }
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
