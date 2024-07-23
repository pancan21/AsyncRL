use common::python::{BoundGetAttrExt, JAX};
use pyo3::{
    pyfunction,
    types::{PyAnyMethods, PyType},
    wrap_pyfunction_bound, Bound, PyAny, PyResult, Python,
};

#[pyfunction]
fn sum_jax_arrays<'py>(a: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    let array_type = JAX.getattr(a.py(), "Array").expect("JAX Array");
    let array_type = array_type
        .downcast_bound::<PyType>(a.py())
        .expect("Is type");

    assert!(a.is_instance(array_type).unwrap());
    let b = a.add(a).unwrap();
    b.call_method0("sum").expect("Jax Array")
}

fn main() {
    pyo3::prepare_freethreaded_python();

    let _ = Python::with_gil(|py| -> PyResult<()> {
        let jax = JAX.get_bound(py);
        let key = jax.getattr_split("random.key")?.call1((1024,))?;
        let x = jax.getattr_split("random.normal")?;
        let y = x.call1((key, (1024, 1024)))?;

        let grad = jax.getattr("grad")?;
        let f = grad.call1((wrap_pyfunction_bound!(sum_jax_arrays)(py)?,))?;
        println!("{}", sum_jax_arrays(&y));
        println!("{}", f.call1((y,))?);
        Ok(())
    });
}
