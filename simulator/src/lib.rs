use common::{
    interfaces::SimulatorInterface,
    messages::{ControlSignalState, ObservableState, Observation, DELAY_DEPTH},
    simulation::{deindex, index, ObservableSimulationState, SimulationConfig, SimulationState},
    vector::Vector,
};
use rayon::prelude::*;

pub struct RustSimulator<T: bytemuck::Pod + num::Float, const DIMS: usize> {
    simulation_states: [SimulationState<T, DIMS>; DELAY_DEPTH + 1],
    observable_substates: [ObservableSimulationState<T, DIMS>; DELAY_DEPTH + 1],
    control_states: [Vec<T>; DELAY_DEPTH + 1],
    offset: usize,
}

fn double_index_sorted<T>(arr: &[T], i1: usize, i2: usize) -> (&T, &T) {
    debug_assert!(i1 < i2);
    let (a, b) = arr.split_at(i2);

    (&a[i1], &b[0])
}

fn double_index_sorted_mut<T>(arr: &mut [T], i1: usize, i2: usize) -> (&mut T, &mut T) {
    debug_assert!(i1 < i2);
    let (a, b) = arr.split_at_mut(i2);

    (&mut a[i1], &mut b[0])
}

fn double_index<T>(arr: &[T], i1: usize, i2: usize) -> (&T, &T) {
    if i1 < i2 {
        double_index_sorted(arr, i1, i2)
    } else if i1 > i2 {
        let (a, b) = double_index_sorted(arr, i2, i1);
        (b, a)
    } else {
        panic!("Expected `i1` != `i2`, but {i1} = i1 = i2 = {i2}");
    }
}

fn double_index_mut<T>(arr: &mut [T], i1: usize, i2: usize) -> (&mut T, &mut T) {
    if i1 < i2 {
        double_index_sorted_mut(arr, i1, i2)
    } else if i1 > i2 {
        let (a, b) = double_index_sorted_mut(arr, i2, i1);
        (b, a)
    } else {
        panic!("Expected `i1` != `i2`, but {i1} = i1 = i2 = {i2}");
    }
}

fn borrow<T: bytemuck::Pod + num::Float, const DIMS: usize>(
    sim: &RustSimulator<T, DIMS>,
    i: usize,
) -> Observation<T, DIMS> {
    Observation {
        time: sim.observable_substates[i].time,
        state: ObservableState {
            size: sim.observable_substates[i].size,
            position: &sim.observable_substates[i].position,
            velocity: &sim.observable_substates[i].velocity,
        },
        controls: ControlSignalState(&sim.control_states[i]),
    }
}

impl<T: bytemuck::Pod + num::Float, const DIMS: usize> RustSimulator<T, DIMS> {
    pub fn new(config: SimulationConfig<T, DIMS>) -> Self {
        let simulation_states = std::array::from_fn(|_| SimulationState::new(config));
        let observable_substates = std::array::from_fn(|_| ObservableSimulationState::new(config));
        let control_states = std::array::from_fn(|_| ControlSignalState::new_vec(config));
        Self {
            simulation_states,
            observable_substates,
            control_states,
            offset: 0,
        }
    }
}

impl<T: bytemuck::Pod + num::Float + Send + Sync, const DIMS: usize> SimulatorInterface<T, DIMS>
    for RustSimulator<T, DIMS>
{
    async fn get_observations<'a>(&'a self) -> [Observation<'a, T, DIMS>; DELAY_DEPTH]
    where
        T: 'a,
    {
        std::array::from_fn(|i| borrow(self, (self.offset + i) % (DELAY_DEPTH + 1)))
    }

    async fn update(&mut self, dt: T) {
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

impl<T: bytemuck::Pod + num::Float, const DIMS: usize> RustSimulator<T, DIMS> {
    fn swap_buffers(
        state: &mut SimulationState<T, DIMS>,
        tmp_acceleration: &mut Box<[Vector<T, DIMS>]>,
    ) {
        std::mem::swap(&mut state.acceleration, tmp_acceleration);
    }

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

    fn compute_forces(
        state: &mut SimulationState<T, DIMS>,
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

    fn par_update_position(state: &mut SimulationState<T, DIMS>, dt: T)
    where
        T: Send + Sync,
    {
        let SimulationState {
            ref mut position,
            ref velocity,
            ..
        } = state;

        position
            .par_iter_mut()
            .zip(velocity.par_iter())
            .for_each(|(p, v)| {
                *p += *v * dt;
            });
    }

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

    fn update_time(
        state: &SimulationState<T, DIMS>,
        next_state: &mut SimulationState<T, DIMS>,
        dt: T,
    ) {
        next_state.time = state.time + dt;
    }

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
                *v += (*a1 + *a2) * dt / (T::one() + T::one());
            });
    }

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
                *v += (*a1 + *a2) * dt / (T::one() + T::one());
            });
    }
}
