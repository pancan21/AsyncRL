use common::{interfaces::SimulatorInterface, vector::Vector, Float};
use smol::{fs::File, io::AsyncWriteExt};

use crate::system::{
    SHOControlSignal, SHOSystemObservation, SHOSystemState, SimpleHarmonicOscillator, DELAY_DEPTH,
};

/// A simple Rust simulator for the [`SimpleHarmonicOscillator`] system.
pub struct SHOSimulator<T: Float> {
    /// The last `[DELAY_DEPTH] + 1` states.
    states: [SHOSystemState<T>; DELAY_DEPTH + 1],
    /// The last `[DELAY_DEPTH] + 1` controls.
    controls: [SHOControlSignal<T>; DELAY_DEPTH + 1],
    /// The offset of the current state.
    offset: usize,

    file: File,
}

impl<T: Float> SHOSimulator<T> {
    /// Creates an instance of [`SHOSimulator`].
    pub fn new(_system: &SimpleHarmonicOscillator<T>) -> Self {
        Self {
            states: [SHOSystemState {
                time: T::zero(),
                position: Vector::zero(),
                velocity: Vector::zero(),
            }; DELAY_DEPTH + 1],
            controls: [SHOControlSignal {
                control: Vector::basis(0),
            }; DELAY_DEPTH + 1],
            offset: 0,
            file: smol::block_on(File::create("./records.csv")).unwrap(),
        }
    }
}

impl<T: Float> SHOSystemState<T> {
    /// Computes the acceleration of the system.
    fn compute_acceleration(&self, stiffness: T, control: SHOControlSignal<T>) -> Vector<T, 2> {
        -self.position * stiffness + control.control
    }
}

impl<T: Float> SimulatorInterface<T, SimpleHarmonicOscillator<T>> for SHOSimulator<T> {
    async fn get_observations(&self) -> Vec<SHOSystemObservation<T>> {
        let mut vec = Vec::with_capacity(DELAY_DEPTH);

        for i in ((self.offset as isize + 1)..(self.offset as isize + 1 + DELAY_DEPTH as isize))
            .map(|i| (i % (DELAY_DEPTH + 1) as isize) as usize)
        {
            vec.push(SHOSystemObservation {
                time: self.states[i].time,
                positions: self.states[i].position,
                controls: self.controls[i],
            })
        }

        vec
    }

    async fn update(
        &mut self,
        system: &SimpleHarmonicOscillator<T>,
        dt: T,
        control_signal: &SHOControlSignal<T>,
    ) {
        let two = T::one() + T::one();

        let next_offset = (self.offset + 1) % (DELAY_DEPTH + 1);
        self.controls[next_offset].clone_from(control_signal);

        let prev_acc = self.states[self.offset]
            .compute_acceleration(system.stiffness, self.controls[self.offset]);

        self.states[next_offset].time = self.states[self.offset].time + dt;
        self.states[next_offset].position = self.states[self.offset].position
            + self.states[self.offset].velocity * dt
            + prev_acc * dt * dt / two;

        let next_acc = self.states[next_offset]
            .compute_acceleration(system.stiffness, self.controls[next_offset]);

        self.states[next_offset].velocity =
            self.states[self.offset].velocity + (prev_acc + next_acc) / two * dt;

        let _ = self
            .file
            .write_all(
                format!(
                    "{:?}, {:?}\n",
                    self.states[self.offset].position[0], self.states[self.offset].position[1]
                )
                .as_bytes(),
            )
            .await;
        self.offset = next_offset;

        if self.offset == 0 {
            self.file.flush().await;
        }

        println!("{}", self.states[self.offset].position.map(|i| i * i).sum())
    }

    fn get_time(&self) -> T {
        self.states[self.offset].time
    }

    async fn get_dynamics_loss(&self) -> T {
        (self.states[self.offset].position.map(|i| i * i).sum() - T::one())
            .powf(T::one() + T::one())
    }
}
