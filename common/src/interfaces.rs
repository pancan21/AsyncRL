#![allow(async_fn_in_trait)]

use crate::{system::System, Float};

/// The interface for an agent driving our dynamical system.
pub trait DriverInterface<T: Float, S: System<T>> {
    /// For a state estimate, computes the control parameters that should be associated with it.
    async fn compute_controls(&self, state_estimate: S::LatentState) -> S::ControlParams;
}

/// The interface for an agent driving our dynamical system.
pub trait GeneratorInterface<T: Float, S: System<T>> {
    /// Sets the parameters that is used for generating the signals.
    async fn set_parameters(&mut self, controls: S::ControlParams, time: T);

    /// Gets the control signal at the given time.
    fn control_signal(&mut self, time: T) -> S::ControlSignal;
}

/// The interface for a simulator for our system.
pub trait SimulatorInterface<T: Float, S: System<T>> {
    /// Gets the last `DELAY_DEPTH` collection of observed states.
    async fn get_observations(&self) -> Vec<S::SystemObservation>;

    /// Updates the state of the system by the given timestep.
    async fn update(&mut self, dt: T, control_signal: &S::ControlSignal);

    /// Gets the current time of the system state.
    fn get_time(&self) -> T;
}

/// The interface for predicting the full state of the system (in some latent space).
pub trait StatePredictionInterface<T: Float, S: System<T>> {
    /// Given a series of observations, predict the full state of the system.
    async fn predict_state(&mut self, observation: &[S::SystemObservation]) -> S::LatentState;
}
