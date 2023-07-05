use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyDict;

use itertools::Itertools;

pub mod mle;

use mle::types::{Grouping, Retrieval};

#[pyfunction]
fn learn_importance(
    py_retrievals: &PyList,
    k: usize,
    learning_rate: f64,
    num_epochs: usize,
    n_jobs: Option<isize>,
    grouping: Option<&PyList>,
) -> PyResult<Vec<f64>> {

    let mut retrievals: Vec<Retrieval> = Vec::with_capacity(py_retrievals.len());
    let mut corpus_size: usize = 0;

    // TODO add helpful error messages
    for py_retrieval in py_retrievals.iter() {
        let retrieved: Vec<usize> = py_retrieval.downcast::<PyDict>()?
            .get_item("retrieved").unwrap()
            .downcast::<PyList>()?.extract().unwrap();

        if !retrieved.is_empty() {
            if *retrieved.iter().max().unwrap() > corpus_size {
                corpus_size = *retrieved.iter().max().unwrap();
            }
        }

        let utility_contributions: Vec<f64> = py_retrieval.downcast::<PyDict>()?
            .get_item("utility_contributions").unwrap()
            .downcast::<PyList>()?.extract().unwrap();

        retrievals.push(Retrieval::new(retrieved, utility_contributions));
    }

    let decoded_grouping = if let Some(py_grouping) = grouping {
        let group_assignments: Vec<usize> = py_grouping.downcast::<PyList>()?.extract().unwrap();
        let num_groups = group_assignments
            .iter()
            .unique() // TODO not sure if hashing things is the fastest option here
            .count();

        Some(Grouping::new(num_groups, group_assignments))
    } else {
        None
    };

    let grouping_reference = decoded_grouping.as_ref();

    let decoded_n_jobs: usize = n_jobs
        .map(|n| if n < 1 { num_cpus::get() } else { n as usize })
        .unwrap_or(1);

    let v = mle::mle_importance(
        retrievals,
        corpus_size + 1,
        grouping_reference,
        k,
        learning_rate,
        num_epochs,
        decoded_n_jobs
    );

    Ok(v)
}

#[pymodule]
fn ragbooster(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(learn_importance, m)?)?;
    Ok(())
}
