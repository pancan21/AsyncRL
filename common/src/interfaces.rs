#![allow(async_fn_in_trait)]

use crate::messages::{
    ControlParameterState, ControlSignalState, Observation, StateTensor, DELAY_DEPTH,
};

/// The interface for an agent driving our dynamical system.
pub trait DriverInterface<T, const DIMS: usize> {
    async fn compute_controls<'a>(
        &'a self,
        state_estimate: StateTensor<T, DIMS>,
    ) -> ControlParameterState<'a, T>
    where
        T: 'a;
}

/// The interface for an agent driving our dynamical system.
pub trait GeneratorInterface<T, const DIMS: usize> {
    async fn set_parameters(&self, controls: ControlParameterState<T>, time: T);
    fn control_parameters(&self, time: T) -> ControlSignalState<T, DIMS>;
}

pub trait SimulatorInterface<T, const DIMS: usize> {
    async fn get_observations<'a>(&'a self) -> [Observation<'a, T, DIMS>; DELAY_DEPTH]
    where
        T: 'a;
    async fn update(&mut self, dt: T);

    fn get_time(&self) -> T;
}

pub trait StatePredictionInterface<T, const DIMS: usize> {
    async fn predict_state(
        &self,
        observation: [Observation<T, DIMS>; DELAY_DEPTH],
    ) -> StateTensor<T, DIMS>;
}
