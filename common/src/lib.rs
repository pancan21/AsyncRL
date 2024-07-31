#![forbid(
    missing_docs,
    clippy::missing_assert_message,
    clippy::missing_docs_in_private_items,
    clippy::missing_asserts_for_indexing,
    // clippy::missing_panics_doc
)]
//! This crate defines the common types, systems, messages, and utilities for the asynchronous RL
//! project.

use std::fmt::{Debug, Display};

use num::NumCast;
use pyo3::{
    types::{PyAnyMethods, PyType},
    FromPyObject, Py, Python, ToPyObject,
};
use python::{BoundGetAttrExt, PythonExt, JAX};

/// Defines the interfaces accessible to different components of the asynchronous RL system:
/// - Generator: Generates the control signal given a set of control parameters
/// - Simulator: Updates the full system state given the control signal
/// - State Estimator: Predicts the full state of the system given the previous observations and
/// the control signals.
/// - RL Agent: Given the (predicted) full state of the system, aims to predict the necessary
/// control parameters to guide the system observations to some target dynamic.
pub mod interfaces;

/// Defines utilities for working with PyO3 for machine learning applications, specificially the
/// JAX library (and friends).
pub mod python;

/// Defines the trait type for a [`System<T: Scalar>`](crate::system::System).
pub mod system;

/// Defines a useful [`Copy`] and [`bytemuck::Pod`]-implementing
/// [`Vector<T, const DIMS: usize>`](crate::vector::Vector) that wraps the array type.
pub mod vector;

/// Defines the [`Rope<T>`] and [`RopeMut<T>`] types that represents references to non-contiguous
/// data.
pub mod rope;

/// The type of [`Float`]
#[derive(Debug, Copy, Clone)]
pub enum FloatType {
    /// 32-bit Float
    F32,
    /// 64-bit Float
    F64,
}

impl FloatType {
    /// Gets JAX dtypes for the float type.
    pub fn jax(self) -> Py<PyType> {
        Python::with_gil_ext(|py| match self {
            FloatType::F32 => JAX
                .bind(py)
                .getattr_split("numpy.float32")
                .unwrap()
                .downcast()
                .unwrap()
                .clone()
                .unbind(),
            FloatType::F64 => JAX
                .bind(py)
                .getattr_split("numpy.float64")
                .unwrap()
                .downcast()
                .unwrap()
                .clone()
                .unbind(),
        })
    }

    /// Gets the python `array.array` specifier for the float type.
    pub fn r#type(self) -> char {
        match self {
            FloatType::F32 => 'f',
            FloatType::F64 => 'd',
        }
    }
}

/// This trait defines the set of floats that have nice computer properties.
pub trait Float:
    num::Float
    + bytemuck::Pod
    + Send
    + Sync
    + Default
    + ToPyObject
    + for<'a> FromPyObject<'a>
    + Debug
    + Display
    + NumCast
{
    /// The type of float represented by this type.
    fn float_type() -> FloatType;
}

impl Float for f32 {
    fn float_type() -> FloatType {
        FloatType::F32
    }
}
impl Float for f64 {
    fn float_type() -> FloatType {
        FloatType::F64
    }
}
