extern crate rand;

use ragbooster::mle as mle;
use mle::types::Retrieval;
use std::time::Instant;
use rand::thread_rng;
use rand::distributions::{Distribution, Bernoulli};

const LEARNING_RATE: f64 = 0.1;
const NUM_STEPS: usize = 10;

fn main() {

    let mut rng = thread_rng();
    let prob_of_right_answer = 0.25;
    let bernoulli = Bernoulli::new(prob_of_right_answer).unwrap();
    let corpus_size = 1000;
    let num_repetitions = 7;

    println!("N,corpus_size,d,k,num_threads,duration");


    #[allow(non_snake_case)]
    //for N in [1_000, 10_000, 100_000, 1_000_000] {
    for N in [1_000_000] {
        for (k, d) in [(10, 50), (20, 100)] {
            let all_retrieved: Vec<Retrieval> = (0..N)
                .map(|_| {
                    let retrieved: Vec<usize> =
                        rand::seq::index::sample(&mut rng, corpus_size, d).into_vec();

                    let utilities: Vec<f64> = (0..d)
                        .map(|_| {
                            if bernoulli.sample(&mut rng) {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    Retrieval::new(retrieved, utilities)
                })
                .collect();

            for num_threads in [1, 2, 4] {
                for _ in 0..num_repetitions {
                    let all_retrieved_copy = all_retrieved.clone();

                    let start_time = Instant::now();
                    let _v = mle::mle_importance(
                        all_retrieved_copy,
                        corpus_size,
                        None,
                        k,
                        LEARNING_RATE,
                        NUM_STEPS,
                        num_threads
                    );

                    let duration = (Instant::now() - start_time).as_millis();
                    println!("{},{},{},{},{},{}", N, corpus_size, d, k, num_threads, duration);
                }
            }
        }
    }
}
