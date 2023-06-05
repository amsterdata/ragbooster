use ragbooster::mle::types::{Grouping, Retrieval};
use ragbooster::mle::v_grouped;


pub struct ValidationSample<V, L> {
    data: V,
    labels: Vec<L>,
}

impl<V, L> ValidationSample<V, L> {
    pub fn new(data: V, labels: Vec<L>) -> Self {
        Self { data, labels }
    }
}

pub struct RetrievalResult<S, L> {
    correct_labels: Vec<L>,
    retrieved_samples: Vec<(S, L)>,
}



pub trait RetrievalModel<V, S, L: Eq> {

    fn query(&self, data: &V) -> Vec<(S, L)>;

    fn corpus_size(&self) -> usize;

    fn label_index(&self, label: &L) -> usize;

    fn sample_index(&self, sample: &S) -> usize;

    fn retrieve(&self, validation_sample: ValidationSample<V, L>) -> RetrievalResult<S, L> {

        let retrieved_samples: Vec<(S, L)> = self.query(&validation_sample.data);

        RetrievalResult {
            correct_labels: validation_sample.labels,
            retrieved_samples
        }
    }

    fn to_retrieved(&self, result: RetrievalResult<S, L>) -> Retrieval {

        let num_samples = result.retrieved_samples.len();

        let mut samples = Vec::with_capacity(num_samples);
        let mut costs = Vec::with_capacity(num_samples);

        for (sample, label) in result.retrieved_samples {
            samples.push(self.sample_index(&sample));
            let cost = if result.correct_labels.contains(&label) { 1.0 } else { 0.0 };
            costs.push(cost);
        }

        Retrieval::new(samples, costs)
    }
}

pub fn mle_importance_from_model<V, S, L: Eq>(
    model: Box<dyn RetrievalModel<V, S, L>>,
    validation_samples: Vec<ValidationSample<V, L>>,
    k: usize,
    num_steps: usize,
    optional_grouping: Option<&Grouping>,
) -> Vec<f64> {

    let precomputed_retrieval_results: Vec<_> = validation_samples.into_iter()
        .map(|validation_sample| {
            let result = model.retrieve(validation_sample);
            model.to_retrieved(result)
        })
        .collect();

    ragbooster::mle::mle_importance(
        precomputed_retrieval_results,
        model.corpus_size(),
        optional_grouping,
        k,
        0.1,
        num_steps,
        1
    )
}


use Question::WhichDBResearcherWonTheTuringAward;
use PossibleAnswer::{MikeStonebraker, JimGray, SSchelter, SGrafberger};
use Website::{Wikipedia, Bing, Fakepedia, Google, DuckduckGo, Liepedia};

#[derive(Eq,PartialEq)]
enum Question { WhichDBResearcherWonTheTuringAward }
#[derive(Eq,PartialEq)]
enum Website { Wikipedia, Bing, Fakepedia, Google, DuckduckGo, Liepedia }
#[derive(Eq,PartialEq)]
enum PossibleAnswer { MikeStonebraker, JimGray, SSchelter, SGrafberger }

struct ToyRetrievalModel {}

impl RetrievalModel<Question, Website, PossibleAnswer> for ToyRetrievalModel {

    fn query(self: &Self, data: &Question) -> Vec<(Website, PossibleAnswer)> {

        if *data == WhichDBResearcherWonTheTuringAward {
            return vec![
                (Wikipedia, MikeStonebraker),
                (Bing, JimGray),
                (Fakepedia, SSchelter),
                (Google, JimGray),
                (DuckduckGo, MikeStonebraker),
                (Liepedia, SGrafberger)
            ];
        } else {
            panic!("Unknown question");
        }
    }

    fn corpus_size(self: &Self) -> usize {
        [Wikipedia, Bing, Fakepedia, Google, DuckduckGo, Liepedia].len()
    }

    fn label_index(self: &Self, label: &PossibleAnswer) -> usize {
        [MikeStonebraker, JimGray, SSchelter, SGrafberger].iter().position(|a| *a == *label)
            .unwrap()
    }

    fn sample_index(self: &Self, sample: &Website) -> usize {
        [Wikipedia, Bing, Fakepedia, Google, DuckduckGo, Liepedia].iter()
            .position(|w| *w == *sample).unwrap()
    }
}

#[test]
fn toy_example() {

    let validation_samples = vec![
        ValidationSample::new(WhichDBResearcherWonTheTuringAward, vec![MikeStonebraker, JimGray]),
    ];

    let model = Box::new(ToyRetrievalModel{});

    let k = 3;

    let corpus_size = model.corpus_size();
    let v = mle_importance_from_model(model, validation_samples, k, 3, None);

    dbg!(&v);

    assert_eq!(v.len(), corpus_size);
    assert!(v[0] > 0.5);
    assert!(v[1] > 0.5);
    assert!(v[2] < 0.5); // Fakepedia gives the wrong answer
    assert!(v[3] > 0.5);
    assert!(v[4] > 0.5);
    assert!(v[5] <= 0.5); // Liepedia gives the wrong answer, but does not get updated here
}

#[test]
fn toy_example_with_groups() {

    let validation_samples = vec![
        ValidationSample::new(WhichDBResearcherWonTheTuringAward, vec![MikeStonebraker, JimGray]),
    ];

    let model = Box::new(ToyRetrievalModel{});

    let k = 3;
    let corpus_size = model.corpus_size();

    let group_assignments = vec![
        0, // Wikipedia
        1, // Bing
        2, // Fakepedia
        1, // Google
        0, // DuckduckGo
        2, // Liepedia
    ];

    let grouping = Grouping::new(3, group_assignments);

    let v = mle_importance_from_model(model, validation_samples, k, 3, Some(&grouping));

    assert_eq!(v.len(), corpus_size);
    assert!(v[0] > 0.5);
    assert!(v[1] > 0.5);
    assert!(v[2] < 0.5); // Fakepedia gives the wrong answer
    assert!(v[3] > 0.5);
    assert!(v[4] > 0.5);
    assert!(v[5] < 0.5); // Liepedia gives the wrong answer

    let v_grouped = v_grouped(&v, &grouping);

    assert_eq!(v_grouped.len(), 3);
    assert!(v_grouped[0] > 0.5);
    assert!(v_grouped[1] > 0.5);
    assert!(v_grouped[2] < 0.5);
}
