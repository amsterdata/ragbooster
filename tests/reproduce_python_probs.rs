use std::io::Read;
use std::fs::File;
use npy::NpyData;
use approx::assert_abs_diff_eq;

use ragbooster::mle::tensors::{DenseTensor, DenseMatrix};
use ragbooster::mle as mle;

fn compare_buffers(rust_buffer: &[f64], python_buffer: &[f64]) {
    assert_eq!(rust_buffer.len(), python_buffer.len());
    for (rust, python) in rust_buffer.iter().zip(python_buffer.iter()) {
        assert_abs_diff_eq!(rust, python, epsilon=0.00000001)
    }
}

fn load_npy(path: &str) -> Vec<f64> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    NpyData::from_bytes(&buf).unwrap().to_vec()
}

#[allow(non_snake_case)]
#[test]
fn compare_IPRP_to_python() {

    let M = 10;
    let K = 3;

    let IP_python = load_npy("test_data/IP_M10_K3_p05.npy");
    let RP_python = load_npy("test_data/RP_M10_K3_p05.npy");

    let p = vec![0.5; M];

    let mut IP = DenseMatrix::new(K + 1, M + 2);
    let mut RP = DenseMatrix::new(K + 1, M + 2);

    mle::prob::compute_prob_from_tensors(&p, K, M, &mut IP, &mut RP);

    compare_buffers(IP.view_buffer(), &IP_python);
    compare_buffers(RP.view_buffer(), &RP_python);

    let IP_python = load_npy("test_data/IP_M10_K3_p025075.npy");
    let RP_python = load_npy("test_data/RP_M10_K3_p025075.npy");

    let p: Vec<f64> = (0..M).map(|i| 0.25 + (i % 2) as f64 * 0.5).collect();

    let mut IP = DenseMatrix::new(K + 1, M + 2);
    let mut RP = DenseMatrix::new(K + 1, M + 2);

    mle::prob::compute_prob_from_tensors(&p, K, M, &mut IP, &mut RP);

    compare_buffers(IP.view_buffer(), &IP_python);
    compare_buffers(RP.view_buffer(), &RP_python);
}

#[allow(non_snake_case)]
#[test]
fn compare_B_to_python() {

    let M = 10;
    let K = 3;
    let E = 5;
    let retrieved_costs: Vec<f64> = vec![1.0; M];
    let distinct_costs: Vec<f64> = vec![0.0, 1.0];

    let B_python = load_npy("test_data/B_M10_K3_p05_E5.npy");

    let p = vec![0.5; M];


    let mut B = DenseTensor::new(K + 1, M + 2, E);

    mle::prob::compute_boundary_set_prob_any_loss_from_tensor_predicated_simd(
        &retrieved_costs,
        &distinct_costs,
        &p,
        K,
        M,
        &mut B
    );

    compare_buffers(B.view_buffer(), &B_python);

    let B_python = load_npy("test_data/B_M10_K3_p025075_E5.npy");

    let p: Vec<f64> = (0..M).map(|i| 0.25 + (i % 2) as f64 * 0.5).collect();

    let mut B = DenseTensor::new(K + 1, M + 2, E);

    mle::prob::compute_boundary_set_prob_any_loss_from_tensor_predicated_simd(
        &retrieved_costs,
        &distinct_costs,
        &p,
        K,
        M,
        &mut B
    );

    compare_buffers(B.view_buffer(), &B_python);
}