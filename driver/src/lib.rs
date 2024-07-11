use pyo3::prelude::PyAnyMethods;
use std::marker::PhantomData;

use common::{interfaces::DriverInterface, python::set_venv_site_packages};
use pyo3::{
    types::{IntoPyDict, PyDict, PyModule},
    Py, PyAny, PyResult, Python,
};
use tqdm::Iter;

pub struct PythonDriver<T, const DIMS: usize> {
    globals: Py<PyDict>,
    locals: Py<PyDict>,
    _phantom: PhantomData<[T; DIMS]>,
}

impl<T, const DIMS: usize> PythonDriver<T, DIMS> {
    pub fn new() -> Self {
        pyo3::prepare_freethreaded_python();

        let [jax, np, model] = Python::with_gil(|py| {
            set_venv_site_packages(py)?;
            let jax = py.import_bound("jax")?;
            let np = py.import_bound("jax.numpy")?;
            let model =
                PyModule::from_code_bound(py, include_str!("model.py"), "model.py", "model")?;

            Ok::<[Py<PyModule>; 3], pyo3::PyErr>([jax.into(), np.into(), model.into()])
        })
        .unwrap();

        let (globals, locals) = Python::with_gil(|py| {
            (
                PyDict::new_bound(py).into(),
                vec![("jax", jax), ("np", np), ("model", model)]
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

    pub fn run_command(&self) -> PyResult<()> {
        let data: Py<PyAny> = Python::with_gil(|py| -> PyResult<_> {
            let model = self
                .locals
                .bind_borrowed(py)
                .get_item("model")?
                .getattr("model")?;
            let mut data = model.call0()?;
            for _ in (0..1000000).tqdm() {
                data = model.call0()?;
            }

            Ok(data.into())
        })?;
        println!("{}", &data);

        Ok(())
    }
}

impl<T, const DIMS: usize> DriverInterface<T, DIMS> for PythonDriver<T, DIMS> {
    async fn compute_controls(
        &self,
        state_estimate: common::messages::StateTensor<T, DIMS>,
    ) -> common::messages::ControlParameterStateOwned<T, DIMS> {
        todo!()
    }
}

impl<T, const DIMS: usize> Default for PythonDriver<T, DIMS> {
    fn default() -> Self {
        Self::new()
    }
}
