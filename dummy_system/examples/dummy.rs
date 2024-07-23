use coordinator::experiment;
use dummy_system::{
    TrivialSystemAgent, TrivialSystemGenerator, TrivialSystemSimulator,
    TrivialSystemState, TrivialSystemStatePredictor,
};
use smol::block_on;

fn main() {
    let simulator = TrivialSystemSimulator {
        states: vec![TrivialSystemState { time: 0. }; 24].into(),
    };
    let generator = TrivialSystemGenerator {
        time: 0.,
        requested_time: 0.,
    };
    let driver = TrivialSystemAgent { time: (0.).into() };
    let state_predictor = TrivialSystemStatePredictor;

    block_on(experiment(
        driver,
        generator,
        simulator,
        state_predictor,
        1e-3,
    ));
}