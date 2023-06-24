// TODO See if we can adjust the visibilities without affecting the integration tests
pub mod prob;
pub mod tensors;
pub mod types;
pub mod gradient;

use crate::mle::types::{Grouping, Retrieval};
use itertools::Itertools;

pub fn mle_importance(
    retrievals: Vec<Retrieval>,
    corpus_size: usize,
    optional_grouping: Option<&Grouping>,
    k: usize,
    learning_rate: f64,
    num_epochs: usize,
    n_jobs: usize,
) -> Vec<f64> {

    let mut v = vec![0.5_f64; corpus_size];

    let max_distinct_retrieved = max_distinct_retrieved(&retrievals);
    let max_distinct_utility_contributions = max_distinct_utility_contributions(&retrievals);

    #[allow(non_snake_case)]
    let N = retrievals.len();

    for _ in 0..num_epochs {
        let g = if n_jobs > 1 {

            rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs)
                .build()
                .unwrap();

            gradient::mle_importance_gradient_parallel(
                &retrievals,
                &v,
                k,
                max_distinct_retrieved,
                max_distinct_utility_contributions,
                N,
                n_jobs
            )
        } else {
            gradient::mle_importance_gradient(
                &retrievals,
                &v,
                k,
                max_distinct_retrieved,
                max_distinct_utility_contributions,
                N,
            )
        };

        for i in 0..v.len() {
            v[i] += learning_rate * g[i];

            // Clipping
            if v[i] > 1.0 {
                v[i] = 1.0;
            } else if v[i] < 0.0 {
                v[i] = 0.0;
            }
        }

        if let Some(grouping) = optional_grouping {
            adjust_for_groups(&mut v, grouping);
        }
    }

    v
}

fn max_distinct_retrieved(retrievals: &[Retrieval]) -> usize {
    retrievals
        .iter()
        .map(|retrieval| retrieval.retrieved.len())
        .max()
        .unwrap()
}

fn max_distinct_utility_contributions(retrievals: &[Retrieval]) -> usize {
    retrievals
        .iter()
        .map(|retrieval| {
            retrieval.utility_contributions.iter()
                // TODO this has to become a parameter!
                .map(|c| (c * 100.0) as i64) // TODO This is ugly and hardcodes a discretization strategy
                .unique() // TODO not sure if hashing things is the fastest option here
                .count()
        })
        .max()
        .unwrap()
}


pub fn v_grouped(v: &[f64], grouping: &Grouping) -> Vec<f64> {

    let mut group_sums = vec![0.0; grouping.num_groups];
    let mut group_counts = vec![0.0; grouping.num_groups];

    for (group, value) in grouping.group_assignments().iter().zip(v.iter()) {
        group_sums[*group] += *value;
        group_counts[*group] += 1.0;
    }

    (0..grouping.num_groups)
        .map(|group| group_sums[group] / group_counts[group])
        .collect()
}

fn adjust_for_groups(v: &mut Vec<f64>, grouping: &Grouping) {

    let v_grouped = v_grouped(v, grouping);

    let assignments = grouping.group_assignments();
    for retrieved_index in 0..v.len() {
        v[retrieved_index] = v_grouped[assignments[retrieved_index]];
    }
}
