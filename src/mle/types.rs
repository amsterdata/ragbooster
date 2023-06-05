

pub struct Grouping {
    pub(crate) num_groups: usize,
    group_per_retrieved: Vec<usize>,
}

impl Grouping {
    pub fn new(num_groups: usize, group_per_retrieved: Vec<usize>) -> Self {
        Self { num_groups, group_per_retrieved }
    }

    pub fn group_assignments(&self) -> &[usize] {
        &self.group_per_retrieved
    }
}

#[derive(Debug, Clone)]
pub struct Retrieval {
    pub(crate) retrieved: Vec<usize>,
    pub(crate) utility_contributions: Vec<f64>,
}

impl Retrieval {

    pub fn new(retrieved: Vec<usize>, utility_contributions: Vec<f64>) -> Self {
        Self { retrieved, utility_contributions }
    }

    pub(crate) fn existence_probabilities(&self, v: &[f64]) -> Vec<f64> {
        self.retrieved
            .iter()
            .map(|r| v[*r])
            .collect()
    }
}
