#[macro_use]
extern crate bencher;
extern crate retrieval_importance;

use bencher::Bencher;
use retrieval_importance::mle::tensors::DenseMatrix;
use retrieval_importance::mle::prob as prob;

benchmark_group!(generate_iprp,
    generate_iprp__no_opt_tiny, generate_iprp__tensors_tiny, generate_iprp__from_tensors_tiny,
    generate_iprp__no_opt, generate_iprp__tensors, generate_iprp__from_tensors,
    generate_iprp__no_opt_larger, generate_iprp__tensors_larger, generate_iprp__from_tensors_larger
);
benchmark_main!(generate_iprp);


#[allow(non_snake_case)]
fn generate_iprp__no_opt_tiny(bench: &mut Bencher) {

    let M = 100;
    let K = 10;
    let p = vec![0.5_f64; M];

    bench.iter(|| {
        bencher::black_box(compute_prob(&p, K, M));
    })
}

#[allow(non_snake_case)]
fn generate_iprp__tensors_tiny(bench: &mut Bencher) {

    let M = 100;
    let K = 10;
    let p = vec![0.5_f64; M];

    bench.iter(|| {
        bencher::black_box(compute_prob_tensors(&p, K, M));
    })
}

#[allow(non_snake_case)]
fn generate_iprp__from_tensors_tiny(bench: &mut Bencher) {

    let M = 100;
    let K = 10;
    let p = vec![0.5_f64; M];
    let mut IP = DenseMatrix::new(K + 1, M + 2);
    let mut RP = DenseMatrix::new(K + 1, M + 2);

    bench.iter(|| {
        bencher::black_box(prob::compute_prob_from_tensors(&p, K, M, &mut IP, &mut RP));
    })
}

#[allow(non_snake_case)]
fn generate_iprp__no_opt(bench: &mut Bencher) {

    let M = 1000;
    let K = 50;
    let p = vec![0.5_f64; M];

    bench.iter(|| {
        bencher::black_box(compute_prob(&p, K, M));
    })
}



#[allow(non_snake_case)]
fn generate_iprp__tensors(bench: &mut Bencher) {

    let M = 1000;
    let K = 50;
    let p = vec![0.5_f64; M];

    bench.iter(|| {
        bencher::black_box(compute_prob_tensors(&p, K, M));
    })
}



#[allow(non_snake_case)]
fn generate_iprp__from_tensors(bench: &mut Bencher) {

    let M = 1000;
    let K = 50;
    let p = vec![0.5_f64; M];
    let mut IP = DenseMatrix::new(K + 1, M + 2);
    let mut RP = DenseMatrix::new(K + 1, M + 2);

    bench.iter(|| {
        bencher::black_box(prob::compute_prob_from_tensors(&p, K, M, &mut IP, &mut RP));
    })
}


#[allow(non_snake_case)]
fn generate_iprp__no_opt_larger(bench: &mut Bencher) {

    let M = 5000;
    let K = 500;
    let p = vec![0.5_f64; M];

    bench.iter(|| {
        bencher::black_box(compute_prob(&p, K, M));
    })
}

#[allow(non_snake_case)]
fn generate_iprp__tensors_larger(bench: &mut Bencher) {

    let M = 5000;
    let K = 500;
    let p = vec![0.5_f64; M];

    bench.iter(|| {
        bencher::black_box(compute_prob_tensors(&p, K, M));
    })
}

#[allow(non_snake_case)]
fn generate_iprp__from_tensors_larger(bench: &mut Bencher) {

    let M = 5000;
    let K = 500;
    let p = vec![0.5_f64; M];
    let mut IP = DenseMatrix::new(K + 1, M + 2);
    let mut RP = DenseMatrix::new(K + 1, M + 2);

    bench.iter(|| {
        bencher::black_box(prob::compute_prob_from_tensors(&p, K, M, &mut IP, &mut RP));
    })
}


#[allow(non_snake_case)]
fn compute_prob(
    p: &Vec<f64>,
    K: usize,
    M: usize
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // TODO We should reuse a preallocated buffer per thread here
    // TODO These should be contiguous arrays to avoid pointer chasing
    // TODO We also waste some space here to not have to compute positions...
    let mut IP = vec![vec![0_f64; M + 2]; K + 1];
    let mut RP = vec![vec![0_f64; M + 2]; K + 1];

    IP[0][0] = 1.0;
    RP[0][M+1] = 1.0;

    // TODO maybe we should manually precompute a 1.0 - p vector? Might double the loads though

    // TODO Make sure the compiler removes bounds checks here
    for j in 1..M+1 {
        IP[0][j] = IP[0][j-1] * (1.0 - p[j-1]);
        for k in 1..K+1 {
            IP[k][j] = IP[k][j-1] * (1.0 - p[j-1]) + IP[k-1][j-1] * p[j-1];
        }
    }

    for j in (1..M+1).rev() {
        RP[0][j] = RP[0][j+1] * (1.0 - p[j-1]);
        for k in 1..K+1 {
            RP[k][j] = RP[k][j+1] * (1.0 - p[j-1]) + RP[k-1][j+1] * p[j-1];
        }
    }

    (IP, RP)
}


#[allow(non_snake_case,unused)]
fn compute_prob_tensors(
    p: &Vec<f64>,
    K: usize,
    M: usize
) -> (DenseMatrix, DenseMatrix) {

    let mut IP = DenseMatrix::new(K + 1, M + 2);
    let mut RP = DenseMatrix::new(K + 1, M + 2);

    IP[[0,0]] = 1.0;
    RP[[0,M+1]] = 1.0;

    for j in 1..M+1 {
        IP[[0,j]] = IP[[0,j-1]] * (1.0 - p[j-1]);
        for k in 1..K+1 {
            IP[[k,j]] = IP[[k,j-1]] * (1.0 - p[j-1]) + IP[[k-1, j-1]] * p[j-1];
        }
    }

    for j in (1..M+1).rev() {
        RP[[0,j]] = RP[[0,j+1]] * (1.0 - p[j-1]);
        for k in 1..K+1 {
            RP[[k,j]] = RP[[k,j+1]] * (1.0 - p[j-1]) + RP[[k-1,j+1]] * p[j-1];
        }
    }

    (IP, RP)
}