use std::marker::PhantomData;

use crate::system::{CoupledHarmonicOscillator, Observation, StateTensor};
use common::{interfaces::StatePredictionInterface, python::set_venv_site_packages, Float};
use pyo3::{
    types::{IntoPyDict, PyAnyMethods, PyDict, PyModule},
    Py, PyAny, PyResult, Python,
};

/// The driver that uses Python with JAX under the hood. This driver is designed for the
/// [`CoupledHarmonicOscillator`].
pub struct PythonStatePredictor<T, const DIMS: usize> {
    /// The set of globally accessible variables.
    globals: Py<PyDict>,
    /// The set of locally accessible variables.
    locals: Py<PyDict>,
    /// Phantom to make `T` and `DIMS` relevant.
    _phantom: PhantomData<[T; DIMS]>,
}

impl<T, const DIMS: usize> PythonStatePredictor<T, DIMS> {
    /// Produces an instance of the [`PythonStatePredictor`].
    pub fn new() -> Self {
        pyo3::prepare_freethreaded_python();

        let [jax, np, state_estimator] = Python::with_gil(|py| {
            set_venv_site_packages(py)?;
            let jax = py.import_bound("jax")?;
            let np = py.import_bound("jax.numpy")?;
            let state_estimator = PyModule::from_code_bound(
                py,
                include_str!("state_estimator.py"),
                "state_estimator.py",
                "state_estimator",
            )?;

            Ok::<[Py<PyModule>; 3], pyo3::PyErr>([jax.into(), np.into(), state_estimator.into()])
        })
        .unwrap_or_else(|_| unreachable!("Python is initialized at the start of the function."));

        let (globals, locals) = Python::with_gil(|py| {
            (
                PyDict::new_bound(py).into(),
                vec![
                    ("jax", jax),
                    ("np", np),
                    ("state_estimator", state_estimator),
                ]
                .into_py_dict_bound(py)
                .into(),
            )
        });

        Self {
            globals,
            locals,
            _phantom: PhantomData,
        }
    }

    /// TODO: REMOVE
    /// This is just a lil toy function
    pub fn run_command(&self) -> PyResult<()> {
        let data: Py<PyAny> = Python::with_gil(|py| -> PyResult<_> {
            let model = self
                .locals
                .bind_borrowed(py)
                .get_item("state_estimator")?
                .getattr("model")?;
            let mut data = model.call0()?;
            for _ in 0..1000000 {
                data = model.call0()?;
            }

            Ok(data.into())
        })?;
        println!("{}", &data);

        Ok(())
    }
}

impl<T: Float, const DIMS: usize> StatePredictionInterface<T, CoupledHarmonicOscillator<T, DIMS>>
    for PythonStatePredictor<T, DIMS>
{
    async fn predict_state(
        &mut self,
        observation: &[Observation<T, DIMS>],
    ) -> StateTensor<T, DIMS> {
        todo!()
    }
}

impl<T, const DIMS: usize> Default for PythonStatePredictor<T, DIMS> {
    fn default() -> Self {
        Self::new()
    }
}
