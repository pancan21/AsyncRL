use common::{interfaces::GeneratorInterface, vector::Vector, Float};

use crate::system::{SHOControlParams, SHOControlSignal, SimpleHarmonicOscillator};

/// The implementer of [`GeneratorInterface`] for the [`SimpleHarmonicOscillator`] system.
pub struct SHOGenerator<T: Float> {
    /// The last time the generator was updated.
    time: T,
    /// The last controls supplied to the generator.
    controls: SHOControlParams<T>,
}

impl<T: Float> SHOGenerator<T> {
    /// Creates an instance of [`SHOGenerator`].
    pub fn new(_system: &SimpleHarmonicOscillator<T>) -> Self {
        Self {
            time: T::zero(),
            controls: SHOControlParams { control: T::zero() },
        }
    }
}

impl<T: Float> GeneratorInterface<T, SimpleHarmonicOscillator<T>> for SHOGenerator<T> {
    async fn set_parameters(&mut self, controls: SHOControlParams<T>, time: T) {
        self.time = time;
        self.controls = controls;
    }

    fn control_signal(&mut self, _time: T) -> SHOControlSignal<T> {
        let (sin, cos) = self.controls.control.sin_cos();
        SHOControlSignal {
            control: Vector::new([sin, cos]) * T::from(1.0).unwrap(),
        }
    }
}
