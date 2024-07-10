use bytemuck::Pod;
use num::Float;

use crate::vector::Vector;

pub struct SimulationState<T: Float + Pod, const DIMS: usize> {
    pub time: T,
    pub size: usize,
    pub stiffness: T,
    pub origin_stiffness: T,
    pub position: Box<[Vector<T, DIMS>]>,
    pub velocity: Box<[Vector<T, DIMS>]>,
    pub acceleration: Box<[Vector<T, DIMS>]>,
}

pub struct ObservableSimulationState<T: Float + Pod, const DIMS: usize> {
    pub time: T,
    pub size: usize,
    pub position: Box<[Vector<T, DIMS>]>,
    pub velocity: Box<[Vector<T, DIMS>]>,
}

fn compute_boundary_size(size: usize) -> usize {
    size * 4 - 4
}

impl<T: Float + Pod, const DIMS: usize> ObservableSimulationState<T, DIMS> {
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        let boundary_size = compute_boundary_size(config.size);
        Self {
            time: T::zero(),
            size: config.size,
            position: vec![Vector::<T, DIMS>::zero(); boundary_size].into_boxed_slice(),
            velocity: vec![Vector::<T, DIMS>::zero(); boundary_size].into_boxed_slice(),
        }
    }
}

impl<T: Float + Pod, const DIMS: usize> SimulationState<T, DIMS> {
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        let SimulationConfig {
            size,
            stiffness,
            origin_stiffness,
        } = config;
        Self {
            size,
            stiffness,
            origin_stiffness,
            time: T::zero(),
            position: vec![Vector::<T, DIMS>::zero(); size * size].into_boxed_slice(),
            velocity: vec![Vector::<T, DIMS>::zero(); size * size].into_boxed_slice(),
            acceleration: vec![Vector::<T, DIMS>::zero(); size * size].into_boxed_slice(),
        }
    }

    pub fn observe(&self, observable: &mut ObservableSimulationState<T, DIMS>) {
        todo!("observe")
    }
}

pub fn deindex<const DIMS: usize>(index: usize, size: usize) -> Vector<usize, DIMS> {
    let modulus = Vector::from_idx(|i| size.pow((DIMS - i - 1) as u32));
    (Vector::broadcast(index) / modulus) % size
}

pub fn index<const DIMS: usize>(index: Vector<usize, DIMS>, size: usize) -> usize {
    let modulus = Vector::from_idx(|i| size.pow((DIMS - i - 1) as u32));
    (index * modulus).sum()
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SimulationConfig<T: Float + Pod, const DIMS: usize> {
    pub size: usize,
    pub stiffness: T,
    pub origin_stiffness: T,
}

impl<T: num::Zero + num::One, const DIMS: usize> Vector<T, DIMS> {
    pub fn basis(idx: usize) -> Self {
        let mut out = Self::zero();
        out[idx].set_one();

        out
    }
}
