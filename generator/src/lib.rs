use bytemuck::Pod;
use common::{
    interfaces::GeneratorInterface,
    messages::{ControlParameterState, ControlSignalState},
    simulation::SimulationConfig,
};
use futures::lock::Mutex;
use num::Float;

pub struct SignalGenerator<T: Float + Pod, const DIMS: usize>(Mutex<(Vec<T>, T)>, Vec<T>);

impl<T: Float + Pod, const DIMS: usize> SignalGenerator<T, DIMS> {
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        SignalGenerator(
            Mutex::new((ControlParameterState::new_vec(config), T::zero())),
            ControlSignalState::new_vec(config),
        )
    }
}

impl<T: Float + Pod, const DIMS: usize> GeneratorInterface<T, DIMS> for SignalGenerator<T, DIMS> {
    async fn set_parameters(&mut self, controls: ControlParameterState<'_, T, DIMS>, time: T) {
        let mut lock = self.0.lock().await;
        lock.0.clear();
        lock.0.copy_from_slice(controls.0);
        lock.1 = time;
    }

    fn control_parameters(&mut self, time: T) -> ControlSignalState<T, DIMS> {
        ControlSignalState(&self.1)
    }
}
