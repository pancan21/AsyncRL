use crate::system::{
    ControlParameterState, ControlSignalState, CoupledHarmonicOscillator, SimulationConfig,
};
use common::{
    interfaces::GeneratorInterface,
    system::System,
    Float,
};
use futures::lock::Mutex;

/// Generates a signal given the last set [`ControlParameterState`] and the time since being set.
/// This is designed for the [`CoupledHarmonicOscillator`] system.
pub struct SignalGenerator<T: Float, const DIMS: usize>(
    Mutex<(ControlParameterState<T, DIMS>, T)>,
    ControlSignalState<T, DIMS>,
);

impl<T: Float, const DIMS: usize> SignalGenerator<T, DIMS> {
    /// Instantiates a new [`SignalGenerator`] based on the given [`SimulationConfig`].
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        SignalGenerator(
            Mutex::new((ControlParameterState::default(config), T::zero())),
            ControlSignalState::default(config),
        )
    }
}

impl<T: Float, const DIMS: usize> GeneratorInterface<T, CoupledHarmonicOscillator<T, DIMS>>
    for SignalGenerator<T, DIMS>
{
    async fn set_parameters(&mut self, controls: ControlParameterState<T, DIMS>, time: T) {
        let mut lock = self.0.lock().await;
        lock.0 .0.clear();
        lock.0 .0.copy_from_slice(&controls.0);
        lock.1 = time;
    }

    fn control_signal(&mut self, _time: T) -> ControlSignalState<T, DIMS> {
        ControlSignalState::new(self.1 .0.clone())
    }
}

/// Produces a constant zero signal.
pub struct DummySignalGenerator;

impl<T: Float, const DIMS: usize> GeneratorInterface<T, CoupledHarmonicOscillator<T, DIMS>>
    for DummySignalGenerator
{
    async fn set_parameters(&mut self, _controls: ControlParameterState<T, DIMS>, _time: T) {}

    fn control_signal(&mut self, _time: T) -> ControlSignalState<T, DIMS> {
        ControlSignalState::new(vec![T::zero(); CoupledHarmonicOscillator::<T, DIMS>::CONTROL_SIGNAL_SIZE])
    }
}
