use common::interfaces::{
    DriverInterface, GeneratorInterface, SimulatorInterface, StatePredictionInterface,
};
use futures::lock::Mutex;

pub async fn experiment<
    T,
    D: DriverInterface<T, DIMS>,
    G: GeneratorInterface<T, DIMS>,
    S: SimulatorInterface<T, DIMS>,
    SP: StatePredictionInterface<T, DIMS>,
    const DIMS: usize,
>(
    driver: D,
    generator: G,
    mut simulator: Mutex<S>,
    state_predictor: SP,
) {
    let observations = simulator.get_mut().get_observations().await;
    let predicted_state = state_predictor.predict_state(observations).await;
    let controls = driver.compute_controls(predicted_state).await;
    generator
        .set_parameters(controls, simulator.get_mut().get_time())
        .await;
}
