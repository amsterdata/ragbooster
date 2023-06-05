#[macro_use]
extern crate bencher;
extern crate retrieval_importance;

use bencher::Bencher;
use retrieval_importance::mle::tensors::DenseTensor;
use retrieval_importance::mle::prob as prob;

benchmark_group!(generate_b, generate_b__no_opt, generate_b__from_tensor,
    generate_b__from_tensor_predicated, generate_b__from_tensor_predicated_simd);
benchmark_main!(generate_b);

const BENCH_M: usize = 1000;
const BENCH_K: usize = 50;
const BENCH_E: usize = 5;

fn generate_bench_data() -> (Vec<f64>, Vec<f64>, Vec<f64>,) {
    let p = vec![0.5_f64; BENCH_M];
    let retrieved_costs: Vec<_> = (0..BENCH_M)
        .map(|i| (i % BENCH_E) as f64 * 1.0 / BENCH_E as f64)
        .collect();

    let distinct_costs: Vec<_> = (0..BENCH_E)
        .map(|i| i as f64 * 1.0 / BENCH_E as f64)
        .collect();

    (p, retrieved_costs, distinct_costs)
}

#[allow(non_snake_case)]
fn generate_b__no_opt(bench: &mut Bencher) {

    let (p, retrieved_costs, distinct_costs) = generate_bench_data();

    bench.iter(|| {
        bencher::black_box(compute_boundary_set_prob_any_loss(
            &retrieved_costs,
            &distinct_costs,
            &p,
            BENCH_K,
            BENCH_M
        ));
    })
}

#[allow(non_snake_case)]
fn generate_b__from_tensor(bench: &mut Bencher) {

    let (p, retrieved_costs, distinct_costs) = generate_bench_data();
    let mut B = DenseTensor::new(BENCH_K + 1, BENCH_M + 2, BENCH_E);

    bench.iter(|| {
        bencher::black_box(compute_boundary_set_prob_any_loss_from_tensor(
            &retrieved_costs,
            &distinct_costs,
            &p,
            BENCH_K,
            BENCH_M,
            &mut B
        ));
    })
}

// TODO this one has a performance regression since we added the zeroing steps...
#[allow(non_snake_case,unused)]
fn generate_b__from_tensor_predicated_simd(bench: &mut Bencher) {

    let (p, retrieved_costs, distinct_costs) = generate_bench_data();
    let mut B = DenseTensor::new(BENCH_K + 1, BENCH_M + 2, BENCH_E);

    bench.iter(|| {
        bencher::black_box(prob::compute_boundary_set_prob_any_loss_from_tensor_predicated_simd(
            &retrieved_costs,
            &distinct_costs,
            &p,
            BENCH_K,
            BENCH_M,
            &mut B
        ));
    })
}

#[allow(non_snake_case,unused)]
fn generate_b__from_tensor_predicated(bench: &mut Bencher) {

    let (p, retrieved_costs, distinct_costs) = generate_bench_data();
    let mut B = DenseTensor::new(BENCH_K + 1, BENCH_M + 2, BENCH_E);

    bench.iter(|| {
        bencher::black_box(compute_boundary_set_prob_any_loss_from_tensor_predicated(
            &retrieved_costs,
            &distinct_costs,
            &p,
            BENCH_K,
            BENCH_M,
            &mut B
        ));
    })
}

#[allow(non_snake_case)]
pub fn compute_boundary_set_prob_any_loss_from_tensor(
    retrieved_costs: &Vec<f64>,
    distinct_costs: &Vec<f64>,
    p: &Vec<f64>,
    K: usize,
    M: usize,
    B: &mut DenseTensor,
) {
    let size_of_e = distinct_costs.len();

    // Required because we reuse un-zeroed memory
    for i in 1..M+2 {
        for e in 0..size_of_e {
            B[[0, i, e]] = 0.0;
        }
    }
    // Required because we reuse un-zeroed memory
    for k in 1..K+1 {
        for e in 0..size_of_e {
            B[[k, M + 1, e]] = 0.0;
        }
    }

    for i in (1..M+1).rev() {
        for k in 1..K+1 {
            for e in 0..size_of_e {
                // This expression should be vectorizable (compute for all e at once)
                B[[k,i,e]] = B[[k,i+1,e]] * (1.0 - p[i-1]) + B[[k-1,i+1,e]] * p[i-1];
                if k==1 && distinct_costs[e] == retrieved_costs[i-1] {
                    B[[k,i,e]] += p[i-1];
                }
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn compute_boundary_set_prob_any_loss_from_tensor_predicated(
    retrieved_costs: &Vec<f64>,
    distinct_costs: &Vec<f64>,
    p: &Vec<f64>,
    K: usize,
    M: usize,
    B: &mut DenseTensor,
) {
    let size_of_e = distinct_costs.len();

    // Required because we reuse un-zeroed memory
    for i in 1..M+2 {
        for e in 0..size_of_e {
            B[[0, i, e]] = 0.0;
        }
    }
    // Required because we reuse un-zeroed memory
    for k in 1..K+1 {
        for e in 0..size_of_e {
            B[[k, M + 1, e]] = 0.0;
        }
    }

    for i in (1..M+1).rev() {
        for e in 0..size_of_e {
            let matches = (distinct_costs[e] == retrieved_costs[i-1]) as i64 as f64;
            B[[1,i,e]] = B[[1,i+1,e]] * (1.0 - p[i-1]) + B[[0,i+1,e]] * p[i-1] + matches * p[i-1];
        }

        for k in 2..K+1 {
            for e in 0..size_of_e {
                B[[k,i,e]] = B[[k,i+1,e]] * (1.0 - p[i-1]) + B[[k-1,i+1,e]] * p[i-1];
            }
        }
    }
}


#[allow(non_snake_case)]
pub fn compute_boundary_set_prob_any_loss(
    retrieved_costs: &Vec<f64>,
    distinct_costs: &Vec<f64>,
    p: &Vec<f64>,
    K: usize,
    M: usize
) -> Vec<Vec<Vec<f64>>> {
    let size_of_e = distinct_costs.len();
    let mut B = vec![vec![vec![0_f64; size_of_e]; M + 2]; K + 1];

    for i in (1..M+1).rev() {
        for k in 1..K+1 {
            for e in 0..size_of_e {
                if k==1 && distinct_costs[e] == retrieved_costs[i-1] {
                    B[k][i][e] += p[i-1];
                }
            }
        }
    }

    B
}

