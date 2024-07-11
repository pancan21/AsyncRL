use pyo3::{types::{PyAnyMethods, PyDict}, PyResult, Python};

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

pub fn set_venv_site_packages(py: Python<'_>) -> PyResult<()> {
    let pydict = PyDict::new_bound(py);
    pydict.set_item("venv", py.eval_bound(&get_venv_location(py), None, None)?)?;

    py.run_bound(
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
        Some(&pydict),
    )?;

    Ok(())
}
