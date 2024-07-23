use std::{ops::Deref, sync::OnceLock};

use itertools::Itertools;
use pyo3::{
    exceptions::PyException,
    types::{PyAnyMethods, PyDict, PyModule},
    Borrowed, Bound, Py, PyAny, PyResult, Python, ToPyObject,
};

/// Tries to find the appropriate Python `sys.path` at runtime.
fn query_shim(py: Python<'_>) -> PyResult<Vec<String>> {
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
    let code = String::from_utf8(output.stdout)
        .expect("Failed to read output of `python -c 'import sys; print(sys.path)'` as utf8");
    let exec = py.eval_bound(&code, None, None)?;
    exec.extract::<Vec<String>>()
        .or(exec.extract::<String>().map(|i| vec![i]))
}

/// Gets the location of the Python virtual environment by checking the following:
/// - The `PY_VENV_LOCATION` environment variable.
/// - The `PY_QUERY_SHIM` environment variable. This calls the [`query_shim`] method.
/// - Attempts to directly add the path `.venv/lib/python{major}.{minor}/site-packages`, where
/// `major` and `minor` are the Python major and minor version numbers respectively.
fn get_venv_location(py: Python<'_>) -> PyResult<Vec<String>> {
    if let Ok(location) = std::env::var("PY_VENV_LOCATION") {
        Ok(location.split(',').map(str::to_string).collect_vec())
    } else if std::env::var_os("PY_QUERY_SHIM").is_some() {
        query_shim(py)
    } else {
        let version = py.version_info();
        let path = format!(
            ".venv/lib/python{}.{}/site-packages",
            version.major, version.minor
        );
        if std::path::Path::new(&path).exists() {
            Ok(vec![path])
        } else {
            Err(PyException::new_err(format!(
                "Did not find virtual environment at {path}"
            )))
        }
    }
}

/// Given a python interpreter instance, modifies the `sys.path` to add the virtual environments if
/// they are not already added.
pub fn set_venv_site_packages(py: Python<'_>) -> PyResult<()> {
    let pydict = PyDict::new_bound(py);
    pydict.set_item("venv", get_venv_location(py)?)?;

    py.run_bound(
        indoc::indoc! {r#"
            import sys
            for v in venv:
                if v not in sys.path:
                    sys.path.append(v)
        "#,
        },
        None,
        Some(&pydict),
    )?;

    Ok(())
}

/// Get a library unbound from the GIL.
pub fn get_library(py: Python<'_>, library: &str) -> PyResult<Py<PyModule>> {
    py.import_bound(library).map(Into::into)
}

/// A lazy-loaded GIL-related object.
pub struct GILLazy<T> {
    /// The initialization function.
    r#fn: fn(Python<'_>) -> T,
    /// The internal oncecell to reference into.
    inner: OnceLock<T>,
}

impl<T> GILLazy<T> {
    /// Make new [`GILLazy<T>`] instance.
    pub const fn new(r#fn: fn(Python<'_>) -> T) -> Self {
        Self {
            r#fn,
            inner: OnceLock::new(),
        }
    }
}

impl<T> Deref for GILLazy<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
            .get_or_init(|| Python::with_gil(|py| (self.r#fn)(py)))
    }
}

impl<T> GILLazy<Py<T>> {
    /// Returns a bound clone of the internal data.
    pub fn get_bound<'a, 'py>(&'a self, py: Python<'py>) -> Borrowed<'a, 'py, T> {
        let _ = set_venv_site_packages(py);
        self.bind_borrowed(py)
    }
}

/// A reference to the JAX library.
pub static JAX: GILLazy<Py<PyModule>> =
    GILLazy::new(|py| get_library(py, "jax").expect("Couldn't load"));

/// A reference to the JAX Numpy library.
pub static NUMPY: GILLazy<Py<PyModule>> =
    GILLazy::new(|py| get_library(py, "numpy").expect("Couldn't load"));

/// Adds some methods to [`Python<'py>`]
pub trait PythonExt {
    /// Injects the [`set_venv_site_packages`] command
    fn with_gil_ext<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R;
}

impl PythonExt for Python<'_> {
    fn with_gil_ext<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        Self::with_gil(|py| {
            let _ = set_venv_site_packages(py).inspect_err(|err| {
                log::info!("Failed to import packages: {}", err);
            });

            f(py)
        })
    }
}

/// Extends objects like [`Py<T>`] to take attributes with periods in them to split them up.
pub trait BoundGetAttrExt<'py> {
    /// Takes in the attribute, splits it by periods and `getattr`'s for each one.
    /// ```notrust
    /// foo.getattr("x.y.z") -> foo.x.y.z
    /// foo.getattr("x") -> foo.x
    /// foo.getattr("") -> foo
    /// ```
    fn getattr_split(&self, attr_name: impl AsRef<str>) -> PyResult<Bound<'py, PyAny>>;

    /// Takes in the attribute, splits it by periods and `getattr`'s for all the elements before
    /// the last period, and `setattr`'s the last element.
    /// ```notrust
    /// foo.setattr("x.y.z", bar) -> foo.x.y.z = bar
    /// foo.setattr("x", bar) -> foo.x = bar
    /// foo.setattr("", bar) -> ! Invalid !
    /// ```
    fn setattr_split(&self, attr_name: impl AsRef<str>, value: impl ToPyObject) -> PyResult<()>;
}

impl<'py, T> BoundGetAttrExt<'py> for Bound<'py, T> {
    fn getattr_split(&self, attr_name: impl AsRef<str>) -> PyResult<Bound<'py, PyAny>> {
        let mut pyobj = self.clone().into_any();
        let attr_name = AsRef::<str>::as_ref(&attr_name);

        if attr_name.is_empty() {
            return Ok(pyobj);
        }

        for i in attr_name.split('.') {
            pyobj = pyobj.getattr(i)?;
        }

        Ok(pyobj)
    }

    fn setattr_split(&self, attr_name: impl AsRef<str>, value: impl ToPyObject) -> PyResult<()> {
        let mut pyobj = self.clone().into_any();
        let mut attr_name = AsRef::<str>::as_ref(&attr_name);
        let split = attr_name.rsplit_once('.');

        if attr_name.is_empty() {
            return Err(PyException::new_err("Received empty attribute string"));
        }

        if let Some((lhs, rhs)) = split {
            attr_name = rhs;
            pyobj = pyobj.getattr_split(lhs)?;
        }

        pyobj.setattr(attr_name, value)
    }
}

impl<'a, 'py, T> BoundGetAttrExt<'py> for Borrowed<'a, 'py, T> {
    fn getattr_split(&self, attr_name: impl AsRef<str>) -> PyResult<Bound<'py, PyAny>> {
        self.as_any().getattr_split(attr_name)
    }

    fn setattr_split(&self, attr_name: impl AsRef<str>, value: impl ToPyObject) -> PyResult<()> {
        self.as_any().setattr_split(attr_name, value)
    }
}

/// Extends objects like [`Bound<'py, T>`] to take attributes with periods in them to split them
/// up.
pub trait UnboundGetAttrExt {
    /// Takes in the attribute, splits it by periods and `getattr`'s for each one.
    /// ```notrust
    /// foo.getattr("x.y.z") -> foo.x.y.z
    /// foo.getattr("x") -> foo.x
    /// foo.getattr("") -> foo
    /// ```
    fn getattr_split(&self, py: Python<'_>, attr_name: impl AsRef<str>) -> PyResult<Py<PyAny>>;

    /// Takes in the attribute, splits it by periods and `getattr`'s for all the elements before
    /// the last period, and `setattr`'s the last element.
    /// ```notrust
    /// foo.setattr("x.y.z", bar) -> foo.x.y.z = bar
    /// foo.setattr("x", bar) -> foo.x = bar
    /// foo.setattr("", bar) -> ! Invalid !
    /// ```
    fn setattr_split(
        &self,
        py: Python<'_>,
        attr_name: impl AsRef<str>,
        value: impl ToPyObject,
    ) -> PyResult<()>;
}

impl<T> UnboundGetAttrExt for Py<T> {
    fn getattr_split(&self, py: Python<'_>, attr_name: impl AsRef<str>) -> PyResult<Py<PyAny>> {
        Ok(self.bind(py).getattr_split(attr_name)?.unbind())
    }

    fn setattr_split(
        &self,
        py: Python<'_>,
        attr_name: impl AsRef<str>,
        value: impl ToPyObject,
    ) -> PyResult<()> {
        self.bind(py).setattr_split(attr_name, value)
    }
}
