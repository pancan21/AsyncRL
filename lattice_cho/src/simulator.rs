use common::{interfaces::SimulatorInterface, vector::Vector, Float};

use crate::system::{
    deindex, index, ControlParameterState, ControlSignalState, CoupledHarmonicOscillator, ObservableSimulationState, ObservableState, Observation, SimulationConfig, SimulationState, DELAY_DEPTH
};
use rayon::prelude::*;

/// A filler trait to make working with the number two easier :)
trait Two: num::Num {
    /// Produces the number two. Under the same constraints as [`num::One::one`].
    fn two() -> Self {
        Self::one() + Self::one()
    }
}

impl<T: num::Num> Two for T {}

/// The [`RustSimulator`] simulates the [`CoupledHarmonicOscillator`] system.
pub struct RustSimulator<T: Float, const DIMS: usize> {
    /// The last `[DELAY_DEPTH] + 1` steps in the system's evolution.
    simulation_states: [SimulationState<T, DIMS>; DELAY_DEPTH + 1],
    /// The last `[DELAY_DEPTH] + 1` steps in the system's observations.
    observable_substates: [ObservableSimulationState<T, DIMS>; DELAY_DEPTH + 1],
    /// The last `[DELAY_DEPTH] + 1` steps in the system's observations.
    control_states: [ControlSignalState<T, DIMS>; DELAY_DEPTH + 1],
    /// The index of the current system state.
    offset: usize,
}

/// Index immutably twice into the array, where the first index parameter is less than the second
/// index parameter.
fn double_index_sorted<T>(arr: &[T], i1: usize, i2: usize) -> (&T, &T) {
    debug_assert!(
        i1 < i2,
        "This method requires that `i1 < i2`, but got i1 = {i1} and i2 = {i2}"
    );
    let (a, b) = arr.split_at(i2);

    (&a[i1], &b[0])
}

/// Index mutably twice into the array, where the first index parameter is less than the second
/// index parameter.
fn double_index_sorted_mut<T>(arr: &mut [T], i1: usize, i2: usize) -> (&mut T, &mut T) {
    debug_assert!(
        i1 < i2,
        "This method requires that `i1 < i2`, but got i1 = {i1} and i2 = {i2}"
    );
    let (a, b) = arr.split_at_mut(i2);

    (&mut a[i1], &mut b[0])
}

/// Index immutably twice into the array, where the first index parameter must be different than
/// the second index parameter.
fn double_index<T>(arr: &[T], i1: usize, i2: usize) -> (&T, &T) {
    match i1.cmp(&i2) {
        std::cmp::Ordering::Less => double_index_sorted(arr, i1, i2),
        std::cmp::Ordering::Greater => {
            let (a, b) = double_index_sorted(arr, i2, i1);
            (b, a)
        }
        std::cmp::Ordering::Equal => {
            panic!("Expected `i1` != `i2`, but {i1} = i1 = i2 = {i2}");
        }
    }
}

/// Index mutably twice into the array, where the first index parameter must be different than the
/// second index parameter.
fn double_index_mut<T>(arr: &mut [T], i1: usize, i2: usize) -> (&mut T, &mut T) {
    match i1.cmp(&i2) {
        std::cmp::Ordering::Less => double_index_sorted_mut(arr, i1, i2),
        std::cmp::Ordering::Greater => {
            let (a, b) = double_index_sorted_mut(arr, i2, i1);
            (b, a)
        }
        std::cmp::Ordering::Equal => {
            panic!("Expected `i1` != `i2`, but {i1} = i1 = i2 = {i2}");
        }
    }
}

impl<T: Float, const DIMS: usize> RustSimulator<T, DIMS> {
    /// Create a new [`RustSimulator<T, DIMS>`] given a config: [`SimulationConfig`].
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        let simulation_states = std::array::from_fn(|_| SimulationState::new(config));
        let observable_substates = std::array::from_fn(|_| ObservableSimulationState::new(config));
        let control_states = std::array::from_fn(|_| ControlSignalState::default(config));
        Self {
            simulation_states,
            observable_substates,
            control_states,
            offset: 0,
        }
    }
}

impl<T: Float + Send + Sync, const DIMS: usize>
    SimulatorInterface<T, CoupledHarmonicOscillator<T, DIMS>> for RustSimulator<T, DIMS>
{
    async fn get_observations(&self) -> Vec<Observation<T, DIMS>> {
        std::array::from_fn::<_, DELAY_DEPTH, _>(|i| {
            let i = (self.offset + i) % (DELAY_DEPTH + 1);
            Observation {
                time: self.observable_substates[i].time,
                state: ObservableState {
                    position: self.observable_substates[i].position.to_vec(),
                    velocity: self.observable_substates[i].velocity.to_vec(),
                },
                controls: self.control_states[i].clone(),
            }
        })
        .to_vec()
    }

    /// The update function here uses [Verlet
    /// integration](https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet)
    async fn update(&mut self, dt: T, control_signal: &ControlSignalState<T, DIMS>) {
        let next_offset = (self.offset + 1) % (DELAY_DEPTH + 1);
        let (tx, rx) = futures::channel::oneshot::channel();

        rayon::scope(|s| {
            let (current_state, next_state) =
                double_index_mut(&mut self.simulation_states, self.offset, next_offset);

            s.spawn(move |_| {
                Self::par_update_position(current_state, dt);
                Self::par_compute_forces(current_state, &mut next_state.acceleration);
                Self::par_update_velocity(current_state, dt, &next_state.acceleration);
                Self::update_time(current_state, next_state, dt);
            });

            tx.send(()).unwrap()
        });

        rx.await.unwrap();
        self.offset += 1;
    }

    fn get_time(&self) -> T {
        self.simulation_states[self.offset].time
    }
}

impl<T: Float, const DIMS: usize> RustSimulator<T, DIMS> {
    /// Swaps the acceleration buffers between [`SimulationState`] and [`Box<\[Vector<T, DIMS>\]>`] by
    /// swapping pointers.
    fn swap_buffers(
        state: &mut SimulationState<T, DIMS>,
        tmp_acceleration: &mut Box<[Vector<T, DIMS>]>,
    ) {
        std::mem::swap(&mut state.acceleration, tmp_acceleration);
    }

    /// Compute the forces on a state in parallel using [`rayon`] and save the accelerations into
    /// the [`Box<\[Vector<T, DIMS>\]>`] reference passed into `tmp_acceleration`.
    fn par_compute_forces(
        state: &SimulationState<T, DIMS>,
        tmp_acceleration: &mut Box<[Vector<T, DIMS>]>,
    ) where
        T: Send + Sync,
    {
        let SimulationState {
            origin_stiffness,
            size,
            stiffness,
            ref position,
            ..
        } = state;

        tmp_acceleration[..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc = -position[i] * *origin_stiffness;

                let idx = deindex::<DIMS>(i, *size);
                for dim in 0..DIMS {
                    if idx[dim] > 0 {
                        let j = index(idx - Vector::<usize, DIMS>::basis(dim), *size);
                        *acc += (position[j] - position[i]) * *stiffness;
                    }
                }
            });
    }

    /// Compute the forces on a state sequentially and save the accelerations into the
    /// [`Box<\[Vector<T, DIMS>\]>`] reference passed into `tmp_acceleration`.
    fn compute_forces(
        state: &SimulationState<T, DIMS>,
        tmp_acceleration: &mut Box<[Vector<T, DIMS>]>,
    ) where
        T: Send + Sync,
    {
        let SimulationState {
            origin_stiffness,
            size,
            stiffness,
            ref position,
            ..
        } = state;

        tmp_acceleration[..]
            .iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc = -position[i] * *origin_stiffness;

                let idx = deindex::<DIMS>(i, *size);
                for dim in 0..DIMS {
                    if idx[dim] > 0 {
                        let j = index(idx - Vector::<usize, DIMS>::basis(dim), *size);
                        *acc += (position[j] - position[i]) * *stiffness;
                    }
                }
            });
    }

    /// Timesteps the positions with a simple first-order update `p(t + dt) = p(t) + dt * v(t) +
    /// (dt^2 / 2) * a(t)` in parallel.
    fn par_update_position(state: &mut SimulationState<T, DIMS>, dt: T)
    where
        T: Send + Sync,
    {
        let SimulationState {
            ref mut position,
            ref velocity,
            ref acceleration,
            ..
        } = state;

        position
            .par_iter_mut()
            .zip(velocity.par_iter().zip(acceleration.par_iter()))
            .for_each(|(p, (v, a))| {
                *p += *v * dt + *a * dt * dt / (T::one() + T::one());
            });
    }

    /// Timesteps the positions with a simple first-order update `p(t + dt) = p(t) + dt * v(t) +
    /// (dt^2 / 2) * a(t)` in sequence.
    fn update_position(state: &mut SimulationState<T, DIMS>, dt: T)
    where
        T: Send + Sync,
    {
        let SimulationState {
            ref mut position,
            ref velocity,
            ..
        } = state;

        position.iter_mut().zip(velocity.iter()).for_each(|(p, v)| {
            *p += *v * dt;
        });
    }

    /// Timesteps the time.
    fn update_time(
        state: &SimulationState<T, DIMS>,
        next_state: &mut SimulationState<T, DIMS>,
        dt: T,
    ) {
        next_state.time = state.time + dt;
    }

    /// Timesteps the positions with a first-order update `v(t + dt) = v(t) + (dt / 2) * (a(t) +
    /// a(t + dt))` in parallel.
    fn par_update_velocity(
        state: &mut SimulationState<T, DIMS>,
        dt: T,
        tmp_acceleration: &[Vector<T, DIMS>],
    ) where
        T: Send + Sync,
    {
        let SimulationState {
            ref mut velocity,
            ref acceleration,
            ..
        } = state;

        velocity
            .par_iter_mut()
            .zip(acceleration.par_iter().zip(tmp_acceleration.par_iter()))
            .for_each(|(v, (a1, a2))| {
                *v += (*a1 + *a2) * dt / T::two();
            });
    }

    /// Timesteps the positions with a first-order update `v(t + dt) = v(t) + (dt / 2) * (a(t) +
    /// a(t + dt))` in sequence.
    fn update_velocity(
        state: &mut SimulationState<T, DIMS>,
        dt: T,
        tmp_acceleration: &mut Box<[Vector<T, DIMS>]>,
    ) where
        T: Send + Sync,
    {
        let SimulationState {
            ref mut velocity,
            ref acceleration,
            ..
        } = state;

        velocity
            .iter_mut()
            .zip(acceleration.iter().zip(tmp_acceleration.iter()))
            .for_each(|(v, (a1, a2))| {
                *v += (*a1 + *a2) * dt / T::two();
            });
    }
}
