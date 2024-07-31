use std::marker::PhantomData;

use common::{
    interfaces::DriverInterface,
    python::{JaxArray, JaxKey, PythonExt},
    system::{DynamicVector, System},
    Float,
};
use pyo3::{
    types::{IntoPyDict, PyAnyMethods, PyModule},
    Bound, Py, PyAny, PyResult, Python, ToPyObject,
};
use smol::lock::Mutex;

use crate::system::{SHOControlParams, SHOLatentState, SimpleHarmonicOscillator};

/// The implementation of [`DriverInterface`] for [`SimpleHarmonicOscillator`]
pub struct SHOAgent<T: Float> {
    /// The object associated with the agent.
    agent: Mutex<Py<PyAny>>,
    /// [`PhantomData`] to support the generic type.
    _phantom: PhantomData<T>,
}

impl<T: Float> SHOAgent<T> {
    /// Creates an instance of [`SHOAgent`].
    pub fn new(key: JaxKey, system: &SimpleHarmonicOscillator<T>) -> Self {
        /// The code in the "sho_agent.py" script.
        const CODE: &str = include_str!("sho_agent.py");

        let agent = Python::with_gil_ext(|py| -> PyResult<Py<PyAny>> {
            let module = PyModule::from_code_bound(py, CODE, "sho_agent.py", "sho_agent")?;

            let agent = module.getattr("SHOAgent")?.getattr("init_state")?.call(
                (),
                Some(
                    &[
                        ("key", key.to_object(py)),
                        (
                            "latent_dimension",
                            SimpleHarmonicOscillator::<T>::LATENT_STATE_SIZE.to_object(py),
                        ),
                        (
                            "control_dimension",
                            SimpleHarmonicOscillator::<T>::CONTROL_PARAMS_SIZE.to_object(py),
                        ),
                        ("gamma", system.gamma.to_object(py)),
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

impl<T: Float> DriverInterface<T, SimpleHarmonicOscillator<T>> for SHOAgent<T> {
    async fn compute_controls(
        &self,
        state_estimate: SHOLatentState<T>,
        dynamics_loss: T,
    ) -> SHOControlParams<T> {
        let mut agent_lock = self.agent.lock().await;
        let array = Python::with_gil_ext(|py| -> PyResult<_> {
            py.check_signals()?;

            let data: JaxArray =
                JaxArray::new_1d(state_estimate.get_rope().into_iter().copied().collect());

            let agent_bound = agent_lock.bind(py);
            let result = agent_bound
                .call_method(
                    "step",
                    (
                        agent_bound,
                        data.to_object(py),
                        (-dynamics_loss).to_object(py),
                    ),
                    None,
                )?
                .extract::<(Bound<PyAny>, Bound<PyAny>)>()?;

            *agent_lock = result.0.unbind();
            Ok(JaxArray::new(result.1.unbind()))
        })
        .unwrap()
        .await
        .into_inner();

        let control = Python::with_gil_ext(|py| {
            array
                .bind(py)
                .call_method0("item")
                .unwrap()
                .extract::<T>()
                .unwrap()
        });
        SHOControlParams { control }
    }
}
