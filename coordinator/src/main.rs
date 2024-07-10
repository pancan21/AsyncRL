use simulator::RustSimulator;

fn main() {
    let config = common::simulation::SimulationConfig {
        size: 128,
        stiffness: 0.1,
        origin_stiffness: 0.3,
    };
    let simulator = RustSimulator::<f64, 5>::new(config);
}
