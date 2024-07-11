#![allow(async_fn_in_trait)]

use crate::messages::{
    ControlParameterState, ControlParameterStateOwned, ControlSignalState, Observation,
    StateTensor, DELAY_DEPTH,
};

/// The interface for an agent driving our dynamical system.
pub trait DriverInterface<T, const DIMS: usize> {
    /// For a state estimate, computes the control parameters that should be associated with it.
    async fn compute_controls(
        &self,
        state_estimate: StateTensor<T, DIMS>,
    ) -> ControlParameterStateOwned<T, DIMS>;
}

/// The interface for an agent driving our dynamical system.
pub trait GeneratorInterface<T, const DIMS: usize> {
    /// Sets the parameters that is used for generating the signals.
    async fn set_parameters(&mut self, controls: ControlParameterState<T, DIMS>, time: T);

    /// Gets the control parameter for the signal at the given time.
    fn control_parameters(&mut self, time: T) -> ControlSignalState<T, DIMS>;
}

/// The interface for a simulator for our system.
pub trait SimulatorInterface<T, const DIMS: usize> {
    /// Gets the last [`DELAY_DEPTH`] collection of observed states.
    async fn get_observations<'a>(&'a self) -> [Observation<'a, T, DIMS>; DELAY_DEPTH]
    where
        T: 'a;

    /// Updates the state of the system by the given timestep.
    async fn update(&mut self, dt: T);

    /// Gets the current time of the system state.
    fn get_time(&self) -> T;
}

/// The interface for predicting the full state of the system (in some latent space).
pub trait StatePredictionInterface<T, const DIMS: usize> {
    /// Given a series of observations, predict the full state of the system.
    async fn predict_state(
        &mut self,
        observation: [Observation<T, DIMS>; DELAY_DEPTH],
    ) -> StateTensor<T, DIMS>;
}
