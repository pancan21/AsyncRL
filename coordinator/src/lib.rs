#![forbid(
    missing_docs,
    clippy::missing_assert_message,
    clippy::missing_docs_in_private_items,
    clippy::missing_asserts_for_indexing,
    clippy::missing_panics_doc
)]
//! This module ties together all the interfaces into an experiment.

use common::{
    interfaces::{
        DriverInterface, GeneratorInterface, SimulatorInterface, StatePredictionInterface,
    },
    system::System,
    Float,
};
use futures::FutureExt;

/// Given a system type, and some [`DriverInterface`], [`GeneratorInterface`],
/// [`SimulatorInterface`], and [`StatePredictionInterface`] implementors (along with a timestep),
/// the experiment control cycle is run.
pub async fn experiment<
    T: Float,
    S: System<T>,
    D: DriverInterface<T, S>,
    G: GeneratorInterface<T, S>,
    SIM: SimulatorInterface<T, S>,
    SP: StatePredictionInterface<T, S>,
>(
    system: &S,
    driver: D,
    mut generator: G,
    mut simulator: SIM,
    mut state_predictor: SP,
    dt: T,
    // TODO: Add some customizable target dynamics into this experiment code.
    // Maybe by means of some given target dynamics loss function?
) {
    let mut current_query = None;
    let mut in_progress = None;
    let future_in_progress =
        |query, dynamics_loss| Box::pin(driver.compute_controls(query, dynamics_loss).fuse());

    let mut i = 0;
    loop {
        i += 1;
        if i % 100 == 0 {
            println!("{i}");
        }

        let observations = simulator.get_observations().await;

        let current_state_estimate = state_predictor.predict_state(&observations).await;
        current_query.replace((current_state_estimate, simulator.get_dynamics_loss().await));

        if in_progress.is_none() {
            if let Some((current_query, dynamics_loss)) = current_query.take() {
                in_progress.replace(future_in_progress(current_query, dynamics_loss));
            }
        }

        if let Some(mut in_progress_future) = in_progress.take() {
            let signal = generator.control_signal(simulator.get_time());
            futures::select! {
                controls = in_progress_future => generator.set_parameters(controls, simulator.get_time()).await,
                _ = simulator.update(system, dt, &signal).fuse() => {
                    in_progress.replace(in_progress_future);
                },
            };
        }
    }
}
