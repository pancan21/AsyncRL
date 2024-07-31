use std::marker::PhantomData;

use common::{
    interfaces::StatePredictionInterface, python::{JaxArray, JaxKey, PythonExt}, system::{DynamicVector, System}, vector::Vector, Float
};
use pyo3::{
    types::{IntoPyDict, PyAnyMethods, PyModule},
    Bound, Py, PyAny, PyResult, Python, ToPyObject,
};
use smol::lock::Mutex;

use crate::system::{SHOLatentState, SHOSystemObservation, SimpleHarmonicOscillator, DELAY_DEPTH};

/// The implementation of [`StatePredictionInterface`] for [`SimpleHarmonicOscillator`]
pub struct SHOStatePredictor<T: Float> {
    /// The object associated with the agent.
    agent: Mutex<Py<PyAny>>,
    /// [`PhantomData`] to support the generic type.
    _phantom: PhantomData<T>,
}

impl<T: Float> SHOStatePredictor<T> {
    /// Creates an instance of [`SHOStatePredictor`].
    pub fn new(key: JaxKey, system: &SimpleHarmonicOscillator<T>) -> Self {
        /// The code in the "sho_state_predictor.py" script.
        const CODE: &str = include_str!("sho_state_predictor.py");

        let agent = Python::with_gil_ext(|py| -> PyResult<Py<PyAny>> {
            let module = PyModule::from_code_bound(
                py,
                CODE,
                "sho_state_predictor.py",
                "sho_state_predictor",
            )?;

            let agent = module
                .getattr("SHOPredictor")?
                .getattr("init_state")?
                .call(
                    (),
                    Some(
                        &[
                            ("key", key.to_object(py)),
                            ("delay_depth", DELAY_DEPTH.to_object(py)),
                            (
                                "observation_dimension",
                                SimpleHarmonicOscillator::<T>::OBSERVABLE_STATE_SIZE.to_object(py),
                            ),
                            (
                                "latent_dimension",
                                SimpleHarmonicOscillator::<T>::LATENT_STATE_SIZE.to_object(py),
                            ),
                        ]
                        .into_py_dict_bound(py),
                    ),
                )?;

            Ok(agent.unbind())
        })
        .unwrap()
        .into();

        Self {
            agent,
            _phantom: PhantomData,
        }
    }
}

impl<T: Float + std::fmt::Debug> StatePredictionInterface<T, SimpleHarmonicOscillator<T>>
    for SHOStatePredictor<T>
{
    async fn predict_state(
        &mut self,
        observation: &[SHOSystemObservation<T>],
    ) -> SHOLatentState<T> {
        let mut agent_lock = self.agent.lock().await;
        let array = Python::with_gil_ext(|py| -> PyResult<JaxArray> {
            let data: JaxArray = JaxArray::new_1d(
                observation
                    .iter()
                    .map(|i| i.get_rope())
                    .reduce(|a, b| a.merge(b))
                    .unwrap()
                    .into_iter()
                    .copied()
                    .collect(),
            );

            let agent_bound = agent_lock.bind(py);
            let result = agent_bound
                .call_method1("step", (agent_bound, data.to_object(py)))?
                .extract::<(Bound<PyAny>, Bound<PyAny>)>()?;

            *agent_lock = result.0.unbind();

            Ok(JaxArray::new(result.1.unbind()))
        })
        .unwrap()
        .await
        .into_inner();

        let latent_representation = Vector::new(Python::with_gil_ext(|py| {
            array
                .bind(py)
                .extract::<[T; 12]>()
                .unwrap()
        }));

        SHOLatentState {
            time: observation.last().unwrap().time,
            latent_representation,
        }
    }
}
