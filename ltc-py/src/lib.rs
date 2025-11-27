use pyo3::prelude::*;

/// Lifetime Clustering (LTC).
#[pyclass]
#[derive(Default)]
struct LifetimeClustering {
    eps: f32,
    labels: Vec<i32>,
    lifetimes: Vec<i32>,
}

#[pymethods]
impl LifetimeClustering {
    /// __init__
    #[new]
    fn new(eps: f32) -> Self {
        Self {
            eps,
            ..Default::default()
        }
    }

    /// Fit model.
    fn fit(&mut self, data: Vec<Vec<f32>>) {
        let (labels, lifetime) = ltc_rs::fit(&data, self.eps);
        self.labels = labels;
        self.lifetimes = lifetime;
    }

    /// Fit model from neighbourhood indices.
    fn fit_with(&mut self, indices: Vec<Vec<usize>>) {
        let (labels, lifetime) = ltc_rs::fit_with(&indices);
        self.labels = labels;
        self.lifetimes = lifetime;
    }

    /// Get labels.
    #[getter]
    fn labels_(&self) -> PyResult<&[i32]> {
        Ok(&self.labels)
    }

    /// Get lifetimes.
    #[getter]
    fn lifetimes_(&self) -> PyResult<&[i32]> {
        Ok(&self.lifetimes)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "ltcpy")]
fn ltcpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LifetimeClustering>()?;
    Ok(())
}
