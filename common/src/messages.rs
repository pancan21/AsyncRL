use crate::{simulation::SimulationConfig, vector::Vector};

/// The number of dimensions of the latent embedding of the system state.
pub const LATENT_SPACE_SHAPE: usize = 1024;

/// The number of previous observations and controls to use to Markovianize the process with the
/// state estimator.
pub const DELAY_DEPTH: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
/// The embedding of the system state in latent space.
pub struct StateTensor<T, const DIMS: usize> {
    pub time: T,
    pub state: Vector<Vector<T, DIMS>, LATENT_SPACE_SHAPE>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
/// The control parameters output by our driver to be fed into our generator.
pub struct ControlParameterState<'a, T>(pub &'a [T]);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
/// The control signal output by our generator.
pub struct ControlSignalState<'a, T, const DIMS: usize>(pub &'a [T]);

impl<T: num::Float + bytemuck::Pod, const DIMS: usize> ControlSignalState<'_, T, DIMS> {
    pub fn new_vec(config: SimulationConfig<T, DIMS>) -> Vec<T> {
        vec![T::zero(); config.size * 4 - 4]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
/// The observable portion of our state.
pub struct ObservableState<'a, T, const DIMS: usize> {
    pub size: usize,
    pub position: &'a [Vector<T, DIMS>],
    pub velocity: &'a [Vector<T, DIMS>],
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(C)]
/// A signal observation of the system and its controls at some time.
pub struct Observation<'a, T, const DIMS: usize> {
    pub time: T,
    pub state: ObservableState<'a, T, DIMS>,
    pub controls: ControlSignalState<'a, T, DIMS>,
}
