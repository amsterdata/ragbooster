use crate::mle::prob;
use crate::mle::tensors::{DenseMatrix, DenseTensor};
use crate::mle::types::Retrieval;
use rayon::prelude::*;

#[allow(non_snake_case)]
pub(crate) fn mle_importance_gradient_parallel(
    D_val: &Vec<Retrieval>, // validation set with labels and ranked retrieved samples
    v: &Vec<f64>, // existence variables
    K: usize, // k of knn-classifier,
    max_distinct_retrieved: usize,
    max_distinct_utility_contributions: usize,
    N: usize,
    n_jobs: usize,
) -> Vec<f64> {

    let M_max = max_distinct_retrieved;
    let E_max = max_distinct_utility_contributions;

    let chunk_size = (D_val.len() / n_jobs) + 1;

    let chunk_gradients: Vec<_> = D_val.par_chunks(chunk_size).map(|retrieval_chunk| {

        let mut g = vec![0.0_f64; v.len()];

        // TODO we allocate per thread/gradient step at the moment,
        // TODO we could also only allocate once per thread
        // TODO we could also compute max_retrieved_samples and max_distinct_labels from the chunk
        let mut IP = DenseMatrix::new(K + 1, M_max + 2);
        let mut RP = DenseMatrix::new(K + 1, M_max + 2);
        let mut B = DenseTensor::new(K + 1, M_max + 2, E_max);

        for retrieval in retrieval_chunk {
            // TODO maybe reuse a buffer here
            let p = retrieval.existence_probabilities(v);
            let s = additive_any_loss_mle_gradient(
                &retrieval.utility_contributions,
                &p,
                K,
                N,
                &mut IP,
                &mut RP,
                &mut B
            );

            for (retrieved_id, contribution) in retrieval.retrieved.iter().zip(s.iter()) {
                g[*retrieved_id] += contribution;
            }
        }
        g
    })
    .collect();

    chunk_gradients
        .into_iter()
        .reduce(|mut sum, gradient| {
            sum.iter_mut().zip(gradient.iter()).for_each(|(s, g)| *s += g);
            sum
        })
        .unwrap()
}

#[allow(non_snake_case)]
pub(crate) fn mle_importance_gradient(
    D_val: &Vec<Retrieval>, // validation set with ranked retrieved samples
    v: &Vec<f64>, // existence variables
    K: usize, // k of knn-classifier,
    max_distinct_retrieved: usize,
    max_distinct_utility_contributions: usize,
    N: usize,
) -> Vec<f64> {

    let M_max = max_distinct_retrieved;
    let E_max = max_distinct_utility_contributions;

    let mut g = vec![0.0_f64; v.len()];

    let mut IP = DenseMatrix::new(K + 1, M_max + 2);
    let mut RP = DenseMatrix::new(K + 1, M_max + 2);
    let mut B =  DenseTensor::new(K + 1, M_max + 2, E_max);

    for retrieval in D_val {
        // TODO maybe reuse a buffer here
        let p = retrieval.existence_probabilities(v);
        let s = additive_any_loss_mle_gradient(
            &retrieval.utility_contributions,
            &p,
            K,
            N,
            &mut IP,
            &mut RP,
            &mut B
        );

        for (retrieved, contribution) in retrieval.retrieved.iter().zip(s.iter()) {
            g[*retrieved] += contribution;
        }
    }

    g
}

/*
def Additive_anyloss_MLE_Gradient_new(v_train, f_train, p, K, M):

    if(len(p)!=M):
        print("p ", len(p))
        print("M ", M)

    s = np.zeros(M)

    IP = np.zeros([K+1, M+2])
    RP = np.zeros([K+1, M+2])
    IP, RP = ComputeProb(p, K, M, IP, RP)

    value_dict = []
    f_dict = []
    for id, i in enumerate(f_train):
        if i not in value_dict:
            value_dict.append(i)
            f_dict.append(f_train[id])
    n_value = len(value_dict)

    B = np.zeros([K+1, M+2, n_value])
    B = ComputeBoundarySetProb_anyloss_new(f_train, value_dict, p, K, M, B)

    for i in range(1, M+1):
        for e in range(0, n_value):
            u_2 = (f_train[i-1] - f_dict[e]) / K
            for j in range(0, K):
                s[i-1] = s[i-1] + u_2 * IP[j][i-1] * B[K-j][i+1][e]
    return s
*/
#[allow(non_snake_case)]
pub fn additive_any_loss_mle_gradient(
    utility_contributions: &[f64],
    p: &[f64],
    K: usize,
    N: usize,
    IP: &mut DenseMatrix,
    RP: &mut DenseMatrix,
    B: &mut DenseTensor
) -> Vec<f64> {

    let num_retrieved = p.len();
    assert_eq!(num_retrieved, utility_contributions.len());

    // TODO we could also reuse a buffer here
    let mut s = vec![0_f64; num_retrieved];

    IP.reuse_as(K + 1, num_retrieved + 2);
    RP.reuse_as(K + 1, num_retrieved + 2);
    prob::compute_prob_from_tensors(p, K, num_retrieved, IP, RP);

    let mut distinct_utility_contributions: Vec<f64> = Vec::new();

    // TODO can this be faster?
    for utility_contribution in utility_contributions {
        if !distinct_utility_contributions.contains(utility_contribution) {
            distinct_utility_contributions.push(*utility_contribution);
        }
    }

    B.reuse_as(K + 1, num_retrieved + 2, distinct_utility_contributions.len());
    prob::compute_boundary_set_prob_any_loss_from_tensor_predicated_simd(
        utility_contributions,
        &distinct_utility_contributions,
        p,
        K,
        num_retrieved,
        B
    );

    for i in 1..num_retrieved +1 {
        let c = utility_contributions[i-1];

        // G_1
        if c != 0.0 {
            let mu_1 = (c as f64 / K as f64) / N as f64;
            for k in 0..K {
                for j in 0..k + 1 {
                    s[i - 1] += mu_1 * IP[[j, i - 1]] * RP[[k - j, i + 1]];
                }
            }
        }

        // G_2
        for e in 0..distinct_utility_contributions.len() {
            let difference = c - distinct_utility_contributions[e];

            if difference != 0.0 {
                let mu_2 = (difference as f64 / K as f64)  / N as f64;
                for j in 0..K {
                    s[i - 1] += mu_2 * IP[[j, i - 1]] * B[[K - j, i + 1, e]];
                }
            }
        }
    }

    s
}
