use std::{collections::VecDeque, sync::Mutex, time::Duration};

use async_std::task::sleep;
use colored::Colorize;
use common::{
    interfaces::{
        DriverInterface, GeneratorInterface, SimulatorInterface, StatePredictionInterface,
    },
    rope::{Rope, RopeMut},
    system::{DynamicVector, System},
};

#[derive(Debug, Clone, Copy)]
pub struct TrivialSystem;

#[derive(Debug, Clone, Copy)]
pub struct TrivialSystemState {
    pub time: f64,
}

impl DynamicVector<f64> for TrivialSystemState {
    fn copy_from_slice(&mut self, v: &[f64]) {
        std::slice::from_mut(&mut self.time).clone_from_slice(v);
    }

    fn get_rope(&self) -> common::rope::Rope<f64> {
        Rope::new(&[std::slice::from_ref(&self.time)])
    }

    fn get_rope_mut(&mut self) -> common::rope::RopeMut<f64> {
        RopeMut::new([std::slice::from_mut(&mut self.time)])
    }
}

impl System<f64> for TrivialSystem {
    const CONTROL_SIGNAL_SIZE: usize = 0;
    const CONTROL_PARAMS_SIZE: usize = 0;
    const LATENT_STATE_SIZE: usize = 1;
    const SYSTEM_STATE_SIZE: usize = 1;
    const OBSERVABLE_STATE_SIZE: usize = 1;

    type SystemConfiguration = ();

    type DynamicsConfiguration = ();

    type SystemState = TrivialSystemState;

    type LatentState = f64;

    type ControlParams = ();

    type ControlSignal = ();

    type SystemObservation = f64;
}

#[derive(Debug, Clone)]
pub struct TrivialSystemSimulator {
    pub states: VecDeque<TrivialSystemState>,
}

#[derive(Debug, Clone, Copy)]
pub struct TrivialSystemGenerator {
    pub time: f64,
    pub requested_time: f64,
}

#[derive(Debug)]
pub struct TrivialSystemAgent {
    pub time: Mutex<f64>,
}

pub struct TrivialSystemStatePredictor;

impl SimulatorInterface<f64, TrivialSystem> for TrivialSystemSimulator {
    async fn update(&mut self, dt: f64, _control_signal: &()) {
        println!("{}", "TrivialSystemSimulator::update".green());
        let new_time = self.get_time() + dt;
        let mut state = self.states.pop_front().unwrap();
        state.time = new_time;

        async_std::task::sleep(Duration::from_millis(100)).await;

        self.states.push_back(state);
    }

    fn get_time(&self) -> f64 {
        self.states.back().unwrap().time
    }

    async fn get_observations(&self) -> Vec<<TrivialSystem as System<f64>>::SystemObservation> {
        println!("TrivialSystemSimulator::get_observations");
        self.states.iter().map(|i| i.time).collect()
    }
}

impl GeneratorInterface<f64, TrivialSystem> for TrivialSystemGenerator {
    async fn set_parameters(
        &mut self,
        _controls: <TrivialSystem as System<f64>>::ControlParams,
        time: f64,
    ) {
        println!("TrivialSystemGenerator::set_parameters @ {time}");
        self.time = time;
    }

    fn control_signal(&mut self, time: f64) -> <TrivialSystem as System<f64>>::ControlSignal {
        println!("TrivialSystemGenerator::control_signal @ {time}");
        self.requested_time = time;
    }
}

impl DriverInterface<f64, TrivialSystem> for TrivialSystemAgent {
    async fn compute_controls(
        &self,
        state_estimate: <TrivialSystem as System<f64>>::LatentState,
    ) -> <TrivialSystem as System<f64>>::ControlParams {
        println!("{}", "TrivialSystemAgent::compute_controls".red());
        *self.time.lock().unwrap() = state_estimate;
        sleep(Duration::from_millis(2000)).await;
    }
}

impl StatePredictionInterface<f64, TrivialSystem> for TrivialSystemStatePredictor {
    async fn predict_state(
        &mut self,
        observation: &[<TrivialSystem as System<f64>>::SystemObservation],
    ) -> <TrivialSystem as System<f64>>::LatentState {
        *observation.last().unwrap()
    }
}
