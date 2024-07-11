use std::marker::PhantomData;

use bytemuck::Pod;
use common::{
    interfaces::{
        DriverInterface, GeneratorInterface, SimulatorInterface, StatePredictionInterface,
    },
    messages::ControlParameterState,
};
use futures::FutureExt;
use num::Float;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ExperimentConfig<T: Float + Pod, const DIMS: usize> {
    dt: T,
    _phantom: PhantomData<[T; DIMS]>,
}

impl<T: Float + Pod, const DIMS: usize> ExperimentConfig<T, DIMS> {
    pub fn build() -> ExperimentConfigBuilder<T, DIMS> {
        ExperimentConfigBuilder {
            dt: None,
            _phantom: PhantomData,
        }
    }
}

pub struct ExperimentConfigBuilder<T: Float + Pod, const DIMS: usize> {
    dt: Option<T>,
    _phantom: PhantomData<[T; DIMS]>,
}

impl<T: Float + Pod, const DIMS: usize> ExperimentConfigBuilder<T, DIMS> {
    pub fn dt(mut self, dt: T) -> Self {
        self.dt.replace(dt);

        self
    }
    pub fn finalize(self) -> ExperimentConfig<T, DIMS> {
        ExperimentConfig {
            dt: self.dt.unwrap_or(T::zero()),
            _phantom: PhantomData,
        }
    }
}

pub async fn experiment<
    T: Float + Pod,
    D: DriverInterface<T, DIMS>,
    G: GeneratorInterface<T, DIMS>,
    S: SimulatorInterface<T, DIMS>,
    SP: StatePredictionInterface<T, DIMS>,
    const DIMS: usize,
>(
    driver: D,
    mut generator: G,
    mut simulator: S,
    mut state_predictor: SP,
    experiment_config: ExperimentConfig<T, DIMS>,
) {
    let mut current_query = None;
    let mut in_progress = None;
    let future_in_progress = |query| driver.compute_controls(query);

    loop {
        let observations = simulator.get_observations().await;

        let current_state_estimate = state_predictor.predict_state(observations).await;
        current_query.replace(current_state_estimate);

        if let Some(current_query) = current_query.take() {
            in_progress.replace(future_in_progress(current_query));
        }

        if let Some(in_progress) = in_progress.take() {
            futures::select! {
                controls = in_progress.fuse() => generator.set_parameters(ControlParameterState::new(&controls.0), simulator.get_time()).await,
                _ = simulator.update(experiment_config.dt).fuse() => {},
            };
        }
    }
}
