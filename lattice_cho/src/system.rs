use std::marker::PhantomData;

use common::{rope::{Rope, RopeMut}, system::{DynamicVector, System}, vector::Vector, Float};

/// The number of dimensions of the latent embedding of the system state.
pub const LATENT_SPACE_SHAPE: usize = 1024;

/// The number of previous observations and controls to use to Markovianize the process with the
/// state estimator.
pub const DELAY_DEPTH: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
/// The embedding of the system state in latent space.
pub struct StateTensor<T, const DIMS: usize> {
    /// The system state time that this latent space tensor represents.
    pub time: T,
    /// The latent space embedding of the system state.
    pub state: Vector<T, LATENT_SPACE_SHAPE>,
}

impl<T: Float, const DIMS: usize> DynamicVector<T> for StateTensor<T, DIMS> {
    fn copy_from_slice(&mut self, v: &[T]) {
        self.state.as_array_mut().copy_from_slice(v);
    }

    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[self.state.as_array()])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([self.state.as_array_mut()])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
/// The control parameters output by our driver to be fed into our generator.
pub struct ControlParameterState<T, const DIMS: usize>(pub Vec<T>, PhantomData<[T; DIMS]>);

impl<T: Float, const DIMS: usize> DynamicVector<T> for ControlParameterState<T, DIMS> {
    fn copy_from_slice(&mut self, v: &[T]) {
        self.0.copy_from_slice(v);
    }

    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[&self.0])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([&mut self.0])
    }
}

impl<T: Float, const DIMS: usize> ControlParameterState<T, DIMS> {
    /// Given data, wraps the data as a [`ControlParameterState`]
    pub fn new(data: Vec<T>) -> Self {
        Self(data, PhantomData)
    }

    /// Given a [`SimulationConfig<T, DIMS>`], produces a [`ControlParameterState<T, DIMS>`] that
    /// has the appropriate shape.
    pub fn default(config: SimulationConfig<T, DIMS>) -> Self {
        Self::new(vec![T::zero(); config.size * 4 - 4])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
/// The control signals output by our driver to be fed into our generator.
pub struct ControlSignalState<T, const DIMS: usize>(pub Vec<T>, PhantomData<[T; DIMS]>);

impl<T: Float, const DIMS: usize> ControlSignalState<T, DIMS> {
    /// Given data, wraps the data as a [`ControlSignalState`]
    pub fn new(data: Vec<T>) -> Self {
        Self(data, PhantomData)
    }

    /// Given a [`SimulationConfig<T, DIMS>`], produces a [`ControlSignalState<T, DIMS>`] that
    /// has the appropriate shape.
    pub fn default(config: SimulationConfig<T, DIMS>) -> Self {
        Self::new(vec![T::zero(); config.size * 4 - 4])
    }
}

impl<T: Float, const DIMS: usize> DynamicVector<T> for ControlSignalState<T, DIMS> {
    fn copy_from_slice(&mut self, v: &[T]) {
        self.0.copy_from_slice(v);
    }

    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[&self.0])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([&mut self.0])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
/// The observable portion of our state.
pub struct ObservableState<T, const DIMS: usize> {
    /// The observable position displacements.
    pub position: Vec<Vector<T, DIMS>>,
    /// The observable velocities.
    pub velocity: Vec<Vector<T, DIMS>>,
}

impl<T: Float, const DIMS: usize> DynamicVector<T> for ObservableState<T, DIMS> {
    fn copy_from_slice(&mut self, v: &[T]) {
        let pos_range = ..self.position.len();
        let vel_range = self.position.len()..;
        self.position
            .copy_from_slice(&bytemuck::cast_slice(v)[pos_range]);
        self.velocity
            .copy_from_slice(&bytemuck::cast_slice(v)[vel_range]);
    }

    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[
            bytemuck::cast_slice(&self.position),
            bytemuck::cast_slice(&self.velocity),
        ])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([
            bytemuck::cast_slice_mut(&mut self.position),
            bytemuck::cast_slice_mut(&mut self.velocity),
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
/// A signal observation of the system and its controls at some time.
pub struct Observation<T, const DIMS: usize> {
    /// The time of the observations.
    pub time: T,
    /// The observable state at the specified time.
    pub state: ObservableState<T, DIMS>,
    /// The control signal applied at the specified time.
    pub controls: ControlSignalState<T, DIMS>,
}

impl<T: Float, const DIMS: usize> DynamicVector<T> for Observation<T, DIMS> {
    fn copy_from_slice(&mut self, v: &[T]) {
        let state_range = ..self.state.get_rope().len();
        let controls_range = self.state.get_rope().len()..;
        self.state.copy_from_slice(&v[state_range]);
        self.controls.copy_from_slice(&v[controls_range]);
    }

    fn get_rope(&self) -> Rope<T> {
        self.state.get_rope().merge(self.controls.get_rope())
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        self.state
            .get_rope_mut()
            .merge(self.controls.get_rope_mut())
    }
}

/// The full state of the CoupleHarmonicOscillator system.
#[derive(Debug, Clone)]
pub struct SimulationState<T: Float, const DIMS: usize> {
    /// The current time in the system.
    pub time: T,
    /// The side-length of the system lattice.
    pub size: usize,
    /// The strength of the coupling between neighboring lattice points.
    pub stiffness: T,
    /// The strength of the coupling between the lattice point and its equilibrium position.
    pub origin_stiffness: T,
    /// The positions of the lattice points.
    pub position: Box<[Vector<T, DIMS>]>,
    /// The velocities of the lattice points.
    pub velocity: Box<[Vector<T, DIMS>]>,
    /// The accelerations of the lattice points.
    pub acceleration: Box<[Vector<T, DIMS>]>,
}

impl<T: Float, const DIMS: usize> Default for SimulationState<T, DIMS> {
    fn default() -> Self {
        Self {
            time: T::zero(),
            size: 0,
            stiffness: T::zero(),
            origin_stiffness: T::zero(),
            position: Box::new([]),
            velocity: Box::new([]),
            acceleration: Box::new([]),
        }
    }
}

impl<T: Float, const DIMS: usize> DynamicVector<T> for SimulationState<T, DIMS> {
    fn copy_from_slice(&mut self, v: &[T]) {
        let s = bytemuck::cast_slice::<_, Vector<T, DIMS>>(v);
        let num_positions = self.position.len();
        let num_velocities = self.velocity.len();
        let num_accelerations = self.acceleration.len();

        let pos_range = 0..num_positions;
        let vel_range = num_positions..(num_positions + num_velocities);
        let acc_range =
            (num_positions + num_velocities)..(num_positions + num_velocities + num_accelerations);

        self.position.copy_from_slice(&s[pos_range]);
        self.velocity.copy_from_slice(&s[vel_range]);
        self.acceleration.copy_from_slice(&s[acc_range]);
    }

    fn get_rope(&self) -> Rope<T> {
        Rope::new(&[
            bytemuck::cast_slice(&self.position),
            bytemuck::cast_slice(&self.velocity),
            bytemuck::cast_slice(&self.acceleration),
        ])
    }

    fn get_rope_mut(&mut self) -> RopeMut<T> {
        RopeMut::new([
            bytemuck::cast_slice_mut(&mut self.position),
            bytemuck::cast_slice_mut(&mut self.velocity),
            bytemuck::cast_slice_mut(&mut self.acceleration),
        ])
    }
}

/// The observable subset of the simulation state. For this system, it is the boundary.
pub struct ObservableSimulationState<T: Float, const DIMS: usize> {
    /// The current time in the system.
    pub time: T,
    /// The side-length of the system lattice.
    pub size: usize,
    /// The positions of the observable lattice points.
    pub position: Box<[Vector<T, DIMS>]>,
    /// The velocities of the observable lattice points.
    pub velocity: Box<[Vector<T, DIMS>]>,
}

/// The configuration of the experiment.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ExperimentConfig<T: Float> {
    /// The time to evolve per system step.
    pub dt: T,
}

/// The [`CoupledHarmonicOscillator`] system describing a `DIMS`-dimensional lattice of coupled
/// harmonic oscillators.
#[derive(Debug, Default)]
pub struct CoupledHarmonicOscillator<T, const DIMS: usize>(PhantomData<[T; DIMS]>);

impl<T: Float, const DIMS: usize> System<T> for CoupledHarmonicOscillator<T, DIMS> {
    const CONTROL_SIGNAL_SIZE: usize = 1; // TODO: Set CONTROL_SIGNAL_SIZE
    const CONTROL_PARAMS_SIZE: usize = 1; // TODO: Set CONTROL_PARAMS_SIZE
    const LATENT_STATE_SIZE: usize = 1; // TODO: Set LATENT_STATE_SIZE
    const SYSTEM_STATE_SIZE: usize = 1; // TODO: Set SYSTEM_STATE_SIZE
    const OBSERVABLE_STATE_SIZE: usize = 1; // TODO: Set OBSERVABLE_STATE_SIZE

    type SystemConfiguration = SimulationConfig<T, DIMS>;
    type DynamicsConfiguration = ExperimentConfig<T>;

    type SystemState = SimulationState<T, DIMS>;

    type LatentState = StateTensor<T, DIMS>;

    type ControlParams = ControlParameterState<T, DIMS>;

    type ControlSignal = ControlSignalState<T, DIMS>;

    type SystemObservation = Observation<T, DIMS>;
}

/// Compute the size of the boundary of the lattice in 2 dimensions.
fn compute_boundary_size(size: usize) -> usize {
    size * 4 - 4
}

impl<T: Float, const DIMS: usize> ObservableSimulationState<T, DIMS> {
    /// Construct a default [`ObservableSimulationState`] from the given configuration.
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        if DIMS != 2 {
            unimplemented!("Haven't implemented this yet!");
        }

        let boundary_size = compute_boundary_size(config.size);
        Self {
            time: T::zero(),
            size: config.size,
            position: vec![Vector::<T, DIMS>::zero(); boundary_size].into_boxed_slice(),
            velocity: vec![Vector::<T, DIMS>::zero(); boundary_size].into_boxed_slice(),
        }
    }
}

impl<T: Float, const DIMS: usize> SimulationState<T, DIMS> {
    /// Construct a default [`SimulationState`] from the given configuration.
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        let SimulationConfig {
            size,
            stiffness,
            origin_stiffness,
        } = config;
        Self {
            size,
            stiffness,
            origin_stiffness,
            time: T::zero(),
            position: vec![Vector::<T, DIMS>::zero(); size.pow(DIMS as u32)].into_boxed_slice(),
            velocity: vec![Vector::<T, DIMS>::zero(); size.pow(DIMS as u32)].into_boxed_slice(),
            acceleration: vec![Vector::<T, DIMS>::zero(); size.pow(DIMS as u32)].into_boxed_slice(),
        }
    }

    /// For a given [`SimulationState`], fill the [`ObservableSimulationState`] with the observable
    /// data of the state.
    pub fn observe(&self, _observable: &mut ObservableSimulationState<T, DIMS>) {
        todo!("observe")
    }
}

/// Given a scalar index into a `DIMS`-dimensional flattened regular array of size `size^DIMS`,
/// compute the vector index.
pub fn deindex<const DIMS: usize>(index: usize, size: usize) -> Vector<usize, DIMS> {
    let modulus = Vector::from_idx(|i| size.pow((DIMS - i - 1) as u32));
    (Vector::broadcast(index) / modulus) % size
}

/// Given a vector index into a `DIMS`-dimensional regular array of where each dimension has size `size`,
/// compute the scalar index.
pub fn index<const DIMS: usize>(index: Vector<usize, DIMS>, size: usize) -> usize {
    let modulus = Vector::from_idx(|i| size.pow((DIMS - i - 1) as u32));
    (index * modulus).sum()
}

/// The configuration for the [`CoupledHarmonicOscillator`] system.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SimulationConfig<T: Float, const DIMS: usize> {
    /// The side-length of the system lattice.
    pub size: usize,
    /// The strength of the coupling between neighboring lattice points.
    pub stiffness: T,
    /// The strength of the coupling between the lattice point and its equilibrium position.
    pub origin_stiffness: T,
}
