use common::python::JaxKey;
use coordinator::experiment;
use sho::{
    driver::SHOAgent, generator::SHOGenerator, simulator::SHOSimulator,
    state_estimator::SHOStatePredictor, system::SimpleHarmonicOscillator,
};
use smol::block_on;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    pyo3::prepare_freethreaded_python();

    let key = JaxKey::key(112045);
    let system = SimpleHarmonicOscillator {
        stiffness: 1.0f32,
        gamma: 1.1,
    };
    let simulator = SHOSimulator::new(&system);
    let generator = SHOGenerator::new(&system);
    let [key, driver_key] = key.split();
    let driver = SHOAgent::new(driver_key, &system);
    let [key, state_predictor_key] = key.split();
    let state_predictor = SHOStatePredictor::new(state_predictor_key, &system);

    block_on(experiment(
        &system,
        driver,
        generator,
        simulator,
        state_predictor,
        1e-2,
    ));

    Ok(())
}
