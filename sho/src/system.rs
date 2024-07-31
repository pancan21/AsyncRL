use common::{
    rope::{Rope, RopeMut},
    system::{DynamicVector, System},
    vector::Vector,
    Float,
};

/// The number of previous observations and controls to use to Markovianize the process with the
/// state estimator.
pub const DELAY_DEPTH: usize = 3;

/// The definition of the 2-D Simple Harmonic Oscillator system.
///
/// The system is given by the following differential equation:
/// $$ \frac{\mathrm{d}^2}{\mathrm{d} t^2}\vec{x} = -k\vec{x} + \vec{F}(t), $$
///
/// where $\vec{x}$ is the current position of the oscillator, $k$ is the stiffness, and $\vec{F}$
/// is the driving force.
#[derive(Debug, Clone, Copy)]
pub struct SimpleHarmonicOscillator<T: Float> {
    /// The stiffness of the Harmonic Oscillator.
    pub stiffness: T,
    /// The reward decay speed.
    pub gamma: T,
}

impl<T: Float> System<T> for SimpleHarmonicOscillator<T> {
    const CONTROL_PARAMS_SIZE: usize = 1;
    const CONTROL_SIGNAL_SIZE: usize = 2;
    const LATENT_STATE_SIZE: usize = 12;
    const SYSTEM_STATE_SIZE: usize = 4;
    const OBSERVABLE_STATE_SIZE: usize = 4;

    type SystemConfiguration = ();

    type DynamicsConfiguration = ();

    type SystemState = SHOSystemState<T>;
    type LatentState = SHOLatentState<T>;
    type ControlParams = SHOControlParams<T>;
    type ControlSignal = SHOControlSignal<T>;
    type SystemObservation = SHOSystemObservation<T>;
}

/// The system state for the [`SimpleHarmonicOscillator`].
#[derive(Debug, Clone, Copy)]
pub struct SHOSystemState<T: Float> {
    /// The current system time.
    pub(crate) time: T,
    /// The position of the oscillator.
    pub(crate) position: Vector<T, 2>,
    /// The velocity of the oscillator.
    pub(crate) velocity: Vector<T, 2>,
}

/// The latent state for the [`SimpleHarmonicOscillator`]
#[derive(Debug, Clone, Copy)]
pub struct SHOLatentState<T: Float> {
    /// The current system time.
    pub(crate) time: T,
    /// The latent representation of the system state.
    pub(crate) latent_representation: Vector<T, 12>,
}

/// The control parameters that are output by a driver agent for the [`SimpleHarmonicOscillator`].
#[derive(Debug, Clone, Copy)]
pub struct SHOControlParams<T: Float> {
    /// The parametrized control signal.
    pub(crate) control: T,
}

/// The control signal that is output by a generator for the [`SimpleHarmonicOscillator`].
#[derive(Debug, Clone, Copy)]
pub struct SHOControlSignal<T: Float> {
    /// The deparametrized control signal. The angle of the force to be applied.
    pub(crate) control: Vector<T, 2>,
}

/// An observation of the current state of the [`SimpleHarmonicOscillator`].
#[derive(Debug, Clone, Copy)]
pub struct SHOSystemObservation<T: Float> {
    /// The current system time.
    pub(crate) time: T,
    /// The current system state.
    pub(crate) positions: Vector<T, 2>,
    /// The current system control signal.
    pub(crate) controls: SHOControlSignal<T>,
}

impl<T: Float> DynamicVector<T> for SHOSystemState<T> {
    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[self.position.as_array(), self.velocity.as_array()])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([self.position.as_array_mut(), self.velocity.as_array_mut()])
    }
}

impl<T: Float> DynamicVector<T> for SHOLatentState<T> {
    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[self.latent_representation.as_array()])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([self.latent_representation.as_array_mut()])
    }
}

impl<T: Float> DynamicVector<T> for SHOControlParams<T> {
    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[std::slice::from_ref(&self.control)])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([std::slice::from_mut(&mut self.control)])
    }
}

impl<T: Float> DynamicVector<T> for SHOControlSignal<T> {
    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[self.control.as_array()])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([self.control.as_array_mut()])
    }
}

impl<T: Float> DynamicVector<T> for SHOSystemObservation<T> {
    fn get_rope(&self) -> Rope<T> {
        self.positions.get_rope().merge(self.controls.get_rope())
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        self.positions
            .get_rope_mut()
            .merge(self.controls.get_rope_mut())
    }
}
