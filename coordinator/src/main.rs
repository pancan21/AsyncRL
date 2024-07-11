use coordinator::{experiment, ExperimentConfig};
use driver::PythonDriver;
use generator::SignalGenerator;
use simulator::RustSimulator;
use state_estimator::PythonStatePredictor;

fn main() {
    let config = common::simulation::SimulationConfig {
        size: 128,
        stiffness: 0.1,
        origin_stiffness: 0.3,
    };
    let experiment_config = ExperimentConfig::<f64, 5>::build().dt(1e-2).finalize();

    let simulator = RustSimulator::new(config);
    let driver = PythonDriver::new();
    let generator = SignalGenerator::new(config);
    let state_predictor = PythonStatePredictor::new();

    futures::executor::block_on(experiment(
        driver,
        generator,
        simulator,
        state_predictor,
        experiment_config,
    ));
}
