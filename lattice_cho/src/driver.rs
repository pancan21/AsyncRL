use pyo3::prelude::PyAnyMethods;
use std::marker::PhantomData;

use crate::system::{ControlParameterState, CoupledHarmonicOscillator, StateTensor};
use common::{interfaces::DriverInterface, python::set_venv_site_packages, system::System, Float};
use pyo3::{
    types::{IntoPyDict, PyDict, PyModule},
    Py, PyAny, PyResult, Python,
};

/// The driver that uses Python with JAX under the hood. This driver is designed for the
/// [`CoupledHarmonicOscillator`].
pub struct PythonDriver<T: Float, const DIMS: usize> {
    /// The set of globally accessible variables.
    globals: Py<PyDict>,
    /// The set of locally accessible variables.
    locals: Py<PyDict>,
    /// Phantom to make `T` and `DIMS` relevant.
    _phantom: PhantomData<[T; DIMS]>,
}

impl<T: Float, const DIMS: usize> PythonDriver<T, DIMS> {
    /// Produces an instance of the [`PythonDriver`].
    pub fn new() -> Self {
        pyo3::prepare_freethreaded_python();

        let [jax, np, driver] = Python::with_gil(|py| {
            set_venv_site_packages(py)?;
            let jax = py.import_bound("jax")?;
            let np = py.import_bound("jax.numpy")?;
            let driver =
                PyModule::from_code_bound(py, include_str!("driver.py"), "driver.py", "driver")?;

            Ok::<[Py<PyModule>; 3], pyo3::PyErr>([jax.into(), np.into(), driver.into()])
        })
        .unwrap_or_else(|_| unreachable!("Python is initialized at the start of the function."));

        let (globals, locals) = Python::with_gil(|py| {
            (
                PyDict::new_bound(py).into(),
                vec![("jax", jax), ("np", np), ("driver", driver)]
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
                .get_item("driver")?
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

impl<T: Float, const DIMS: usize> DriverInterface<T, CoupledHarmonicOscillator<T, DIMS>>
    for PythonDriver<T, DIMS>
{
    async fn compute_controls(
        &self,
        state_estimate: StateTensor<T, DIMS>,
    ) -> ControlParameterState<T, DIMS> {
        todo!()
    }
}

impl<T: Float, const DIMS: usize> Default for PythonDriver<T, DIMS> {
    fn default() -> Self {
        Self::new()
    }
}
