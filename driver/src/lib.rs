use std::marker::PhantomData;

use common::interfaces::DriverInterface;
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

fn query_shim() -> String {
    let handle = std::process::Command::new("python")
        .arg("-c")
        .arg("import sys; print(sys.path)")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to run `python -c 'import sys; print(sys.path)'`");

    let output = handle
        .wait_with_output()
        .expect("failed to wait on `python -c 'import sys; print(sys.path)'`");
    if !output.status.success() {
        panic!(
            "`python -c 'import sys; print(sys.path)'` command failed with status {}",
            output.status
        );
    }
    String::from_utf8(output.stdout)
        .expect("Failed to read output of `python -c 'import sys; print(sys.path)'` as utf8")
}

fn get_venv_location(py: Python<'_>) -> String {
    if let Ok(location) = std::env::var("PY_VENV_LOCATION") {
        location
    } else if std::env::var_os("PY_QUERY_SHIM").is_some() {
        query_shim()
    } else {
        let version = py.version_info();
        format!(
            "'.venv/lib/python{}.{}/site-packages'",
            version.major, version.minor
        )
    }
}

fn set_venv_site_packages(py: Python<'_>) -> PyResult<()> {
    let pydict = PyDict::new(py);
    pydict.set_item("venv", py.eval(&get_venv_location(py), None, None)?)?;

    py.run(
        indoc::indoc! {r#"
            import sys
            if isinstance(venv, str):
                venv = [venv]

            for v in venv:
                if v not in sys.path:
                    sys.path.append(v)
        "#,
        },
        None,
        Some(pydict),
    )?;

    Ok(())
}

impl<T, const DIMS: usize> PythonDriver<T, DIMS> {
    pub fn new() -> Self {
        pyo3::prepare_freethreaded_python();

        let [jax, np, model] = Python::with_gil(|py| {
            set_venv_site_packages(py)?;
            let jax = py.import("jax")?;
            let np = py.import("jax.numpy")?;
            let model = PyModule::from_code(py, include_str!("model.py"), "model.py", "model")?;

            Ok::<[Py<PyModule>; 3], pyo3::PyErr>([jax.into(), np.into(), model.into()])
        })
        .unwrap();

        let (globals, locals) = Python::with_gil(|py| {
            (
                PyDict::new(py).into(),
                vec![("jax", jax), ("np", np), ("model", model)]
                    .into_py_dict(py)
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
                .as_ref(py)
                .get_item("model")
                .ok()
                .flatten()
                .unwrap()
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
    async fn compute_controls<'a>(
        &'a self,
        state_estimate: common::messages::StateTensor<T, DIMS>,
    ) -> common::messages::ControlParameterState<'a, T>
    where
        T: 'a,
    {
        todo!()
    }
}

impl<T, const DIMS: usize> Default for PythonDriver<T, DIMS> {
    fn default() -> Self {
        Self::new()
    }
}
