use ragbooster::mle as mle;
use num_cpus;
use std::time::Instant;
use itertools::Itertools;

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
const NUM_STEPS: usize = 100;

fn wikifact(questions_file: &str, n_jobs: usize) -> Vec<f64> {
    eprintln!("Reading {questions_file}");
    let (all_retrieved, website_indexer) = read_qa_json(questions_file).unwrap();

    let corpus_size = website_indexer.num_observed_strings();

    eprintln!("Found {} questions and {} websites...", all_retrieved.len(), corpus_size);


    eprintln!("Computing MLE importance with k={K} and {n_jobs} thread(s) for {NUM_STEPS} step(s)");

    let start_time = Instant::now();
    let v = mle::mle_importance(
        all_retrieved,
        corpus_size,
        None,
        K,
        LEARNING_RATE,
        NUM_STEPS,
        n_jobs
    );
    let duration = (Instant::now() - start_time).as_millis();

    let num_updated = v.len();

    eprintln!("Computed importance for {num_updated} websites in {duration}ms");

    let sorted_indexes_for_v: Vec<usize> = v
        .iter()
        .enumerate()
        .sorted_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .collect();

    let index_to_website = website_indexer.create_reverse_index();

    eprintln!("Three most important websites:");
    sorted_indexes_for_v.iter().rev().take(5).for_each(|index | {
        let website = index_to_website.get(index).unwrap();
        let v_of_website = v[*index];
        eprintln!("  {website} {v_of_website}");
    });

    eprintln!("Three least important websites:");
    sorted_indexes_for_v.iter().take(5).rev().for_each(|index | {
        let website = index_to_website.get(index).unwrap();
        let v_of_website = v[*index];
        eprintln!("  {website} {v_of_website}");
    });

    v
}

#[test]
fn wikifact_authors() {
    wikifact("test_data/wikifact/author.jsonl", 1);
}

#[test]
fn wikifact_place_of_birth() {
    wikifact("test_data/wikifact/place_of_birth.jsonl", 1);
}

#[test]
fn wikifact_currency() {
    wikifact("test_data/wikifact/currency.jsonl", 1);
}

#[test]
fn parallelism() {
    let v_single_threaded = wikifact("test_data/wikifact/author.jsonl", 1);
    let v_parallel = wikifact("test_data/wikifact/author.jsonl", num_cpus::get());

    let norm_of_difference = v_single_threaded
        .iter().zip(v_parallel)
        .fold(0.0, |acc, (vs, vp)| acc + (vs - vp) * (vs - vp))
        .sqrt();

    assert!(norm_of_difference < 0.0000001);
}


fn wikifact_grouped(
    questions_file: &str,
    group_file: &str,
    n_jobs: usize,
    learning_rate: f64,
    num_steps: usize,
) -> Vec<f64> {
    eprintln!("Reading {questions_file}");
    let (all_retrieved, website_indexer) = read_qa_json(questions_file).unwrap();

    let corpus_size = website_indexer.num_observed_strings();

    eprintln!("Found {} questions and {} websites...", all_retrieved.len(), corpus_size);

    let website_to_id = website_indexer.create_index();



    let (grouping, group_indexer) = read_group_json(group_file, &website_to_id).unwrap();

    eprintln!("Computing MLE importance with k={K} and {n_jobs} thread(s) for {NUM_STEPS} step(s)");

    let start_time = Instant::now();
    let v = mle::mle_importance(
        all_retrieved,
        corpus_size,
        Some(&grouping),
        K,
        learning_rate,
        num_steps,
        n_jobs
    );
    let duration = (Instant::now() - start_time).as_millis();

    let num_updated = v.len();

    eprintln!("Computed importance for {num_updated} websites in {duration}ms");

    let v_grouped = mle::v_grouped(&v, &grouping);

    let sorted_indexes_for_v_grouped: Vec<usize> = v_grouped
        .iter()
        .enumerate()
        .sorted_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .collect();

    let index_to_group = group_indexer.create_reverse_index();

    eprintln!("Three most important groups:");
    sorted_indexes_for_v_grouped.iter().rev().take(5).for_each(|index | {
        let group = index_to_group.get(index).unwrap();
        let v_of_group = v_grouped[*index];
        eprintln!("  {group} {v_of_group}");
    });

    eprintln!("Three least important groups:");
    sorted_indexes_for_v_grouped.iter().take(5).rev().for_each(|index | {
        let group = index_to_group.get(index).unwrap();
        let v_of_group = v_grouped[*index];
        eprintln!("  {group} {v_of_group}");
    });

    v
}

#[test]
fn wikifact_grouped_author() {
    wikifact_grouped(
        "test_data/wikifact/author.jsonl",
        "test_data/wikifact/author_websites_by_domain.jsonl",
        1,
        LEARNING_RATE,
        NUM_STEPS);
}

#[test]
fn wikifact_grouped_author_no_nan() {
    let v = wikifact_grouped(
        "test_data/wikifact/author.jsonl",
        "test_data/wikifact/author_websites_by_domain.jsonl",
        num_cpus::get(),
        0.9,
        1000);

    for v_i in &v {
        assert!(!v_i.is_nan());
    }
}

#[test]
fn wikifact_grouped_place_of_birth() {
    wikifact_grouped(
        "test_data/wikifact/place_of_birth.jsonl",
        "test_data/wikifact/place_of_birth_websites_by_domain.jsonl",
        1,
        LEARNING_RATE,
        NUM_STEPS);
}

#[test]
fn wikifact_grouped_currency() {
    wikifact_grouped(
        "test_data/wikifact/currency.jsonl",
        "test_data/wikifact/currency_websites_by_domain.jsonl",
        1,
        LEARNING_RATE,
        NUM_STEPS);
}

#[test]
fn parallelism_grouped() {
    let v_single_threaded = wikifact_grouped(
        "test_data/wikifact/author.jsonl",
        "test_data/wikifact/author_websites_by_domain.jsonl",
        1,
        LEARNING_RATE,
        NUM_STEPS);

    let v_parallel = wikifact_grouped(
        "test_data/wikifact/author.jsonl",
        "test_data/wikifact/author_websites_by_domain.jsonl",
        num_cpus::get(),
        LEARNING_RATE,
        NUM_STEPS);

    let norm_of_difference = v_single_threaded
        .iter().zip(v_parallel)
        .fold(0.0, |acc, (vs, vp)| acc + (vs - vp) * (vs - vp))
        .sqrt();

    assert!(norm_of_difference < 0.0000001);
}
