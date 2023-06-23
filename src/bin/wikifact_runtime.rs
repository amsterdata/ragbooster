use ragbooster::mle as mle;
use std::time::Instant;

use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use std::collections::BTreeSet;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use mle::types::{Grouping, Retrieval};


#[derive(Serialize, Deserialize, Debug)]
struct QuestionAnswering {
    question: String,
    correct_answers: Vec<String>,
    retrieved_websites: Vec<String>,
    retrieved_answers: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Group {
    name: String,
    elements: Vec<String>,
}

#[derive(Default)]
pub struct StringIndexer {
    distinct_strings: BTreeSet<String>,
}

impl StringIndexer {
    pub fn new() -> Self {
        Self { distinct_strings: BTreeSet::new() }
    }

    pub fn observe_all(&mut self, strs: &[String]) {
        for a_str in strs {
            self.observe(a_str.clone());
        }
    }

    pub fn num_observed_strings(&self) -> usize {
        self.distinct_strings.len()
    }

    pub fn observe(&mut self, str: String) {
        self.distinct_strings.insert(str);
    }

    pub fn create_index(&self) -> HashMap<String, usize> {
        let mut index = HashMap::new();
        for (id, website) in self.distinct_strings.iter().enumerate() {
            index.insert((*website).to_owned(), id);
        }
        index
    }

    pub fn create_reverse_index(&self) -> HashMap<usize, String> {
        let mut index = HashMap::new();
        for (id, website) in self.distinct_strings.iter().enumerate() {
            index.insert(id, (*website).to_owned());
        }
        index
    }
}


pub fn read_qa_json(path: &str) -> io::Result<(Vec<Retrieval>, StringIndexer)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut inputs: Vec<QuestionAnswering> = Vec::new();

    for line in reader.lines() {
        let input: QuestionAnswering = serde_json::from_str(&line?)?;
        inputs.push(input);
    }

    let mut website_indexer = StringIndexer::new();

    for (line, input) in inputs.iter().enumerate() {

        assert!(!input.correct_answers.is_empty(), "correct_answers empty in line {}", line);
        assert!(!input.retrieved_websites.is_empty(), "no retrieved websites in line {}", line);
        assert_eq!(input.retrieved_websites.len(), input.retrieved_answers.len(),
                   "inconsistent number of websites and answers in line {}", line);

        website_indexer.observe_all(&input.retrieved_websites);
    }

    let website_index = website_indexer.create_index();

    let mut all_retrieved = Vec::with_capacity(inputs.len());

    for input in &inputs {

        let samples: Vec<usize> = input.retrieved_websites.iter()
            .map(|website| *website_index.get(website).unwrap())
            .collect();

        let costs: Vec<f64> = input.retrieved_answers.iter()
            .map(|answer| {
                if input.correct_answers.contains(answer) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let retrieved = Retrieval::new(samples, costs);

        all_retrieved.push(retrieved);
    }

    Ok((all_retrieved, website_indexer))
}

pub fn read_group_json(
    path: &str,
    element_index: &HashMap<String, usize>
) -> io::Result<(Grouping, StringIndexer)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut group_name_indexer = StringIndexer::new();
    let mut groups: Vec<Group> = Vec::new();

    for line in reader.lines() {
        let group: Group = serde_json::from_str(&line?)?;
        group_name_indexer.observe(group.name.clone());
        groups.push(group);
    }

    let group_name_index = group_name_indexer.create_index();
    let mut group_per_samples = vec![0; element_index.len()];

    let mut num_elements_mapped: usize = 0;
    for group in groups {
        let group_id = group_name_index.get(&group.name).unwrap();
        for element in group.elements {
            let element_id = element_index.get(&element).unwrap();
            group_per_samples[*element_id] = *group_id;
            num_elements_mapped += 1;
        }
    }
    assert_eq!(
        num_elements_mapped,
        element_index.len(),
        "Group information missing for {} elements", (element_index.len() - num_elements_mapped));

    Ok((Grouping::new(group_name_index.len(), group_per_samples), group_name_indexer))
}

const K: usize = 10;
const LEARNING_RATE: f64 = 0.1;
const NUM_STEPS: usize = 10;

fn wikifact(questions_file: &str, n_jobs: usize) -> u128 {
    let (all_retrieved, website_indexer) = read_qa_json(questions_file).unwrap();
    let corpus_size = website_indexer.num_observed_strings();

    let start_time = Instant::now();
    let _v = mle::mle_importance(
        all_retrieved,
        corpus_size,
        None,
        K,
        LEARNING_RATE,
        NUM_STEPS,
        n_jobs
    );
    let duration = (Instant::now() - start_time).as_millis();

    //eprintln!("{}", v.len());

    return duration
}

// RUSTFLAGS="-C target-cpu=native" cargo run --release --bin wikifact_runtime
fn main() {

    let files = [
        "test_data/wikifact/author.jsonl",
        "test_data/wikifact/place_of_birth.jsonl",
        "test_data/wikifact/currency.jsonl"
    ];

    let threads = [1, 2, 4];

    let num_repetitions = 7;

    for file in files {
        for num_threads in threads {
            for _ in 0..num_repetitions {
                let duration = wikifact(file, num_threads);
                println!("{},{},{}", file, num_threads, duration);
            }
        }
    }
}
