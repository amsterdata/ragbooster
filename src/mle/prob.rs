use crate::mle::tensors::{DenseMatrix, DenseTensor};

/*
IP = np.zeros([K+1, M+2])
RP = np.zeros([K+1, M+2])
def ComputeProb(p, K, M, IP, RP):
    IP[0][0] = 1
    RP[0][M+1] = 1

    for j in range(1, M+1):
        IP[0][j] = IP[0][j-1] * (1 - p[j-1])
        for k in range(1,K+1):
            IP[k][j] = IP[k][j-1] * (1 - p[j-1]) + IP[k-1][j-1] * p[j-1]

    for j in range(M, 0, -1):
        RP[0][j] = RP[0][j+1] * (1 - p[j-1])
        for k in range(1,K+1):
            RP[k][j] = RP[k][j+1] * (1 - p[j-1]) + RP[k-1][j+1] * p[j-1]

    return IP, RP
*/
#[allow(non_snake_case,unused)]
pub fn compute_prob_from_tensors(
    p: &[f64],
    K: usize,
    M: usize,
    IP: &mut DenseMatrix,
    RP: &mut DenseMatrix,
) {
    IP[[0,0]] = 1.0;
    RP[[0,M+1]] = 1.0;

    // Required because we reuse un-zeroed memory
    for k in 1..K+1 {
        IP[[k,0]] = 0.0;
        RP[[k,M+1]] = 0.0;
    }
    // Required because we reuse un-zeroed memory
    for j in 1..M {
        IP[[0,j]] = 0.0;
    }

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
}

/*
value_dict = []
f_dict = []
for id, i in enumerate(v_train):
    if i not in value_dict:
        value_dict.append(i)
        f_dict.append(f_train[id])
n_value = len(value_dict)

B = np.zeros([K+1, M+2, n_value])

def ComputeBoundarySetProb_anyloss(v_train, value_dict, p, K, M, B):
    n_value = len(value_dict)
    for i in range(M, 0, -1):
        for k in range(1, K+1):
            for e in range(0, n_value):
                B[k][i][e] = B[k][i+1][e] * (1 - p[i-1]) + B[k-1][i+1][e] * p[i-1]
                if (k==1) & (value_dict[e] == v_train[i-1]):
                    B[k][i][e] += p[i-1]
    return B

def ComputeBoundarySetProb_anyloss_new(f_train, value_dict, p, K, M, B):
    n_value = len(value_dict)
    for i in range(M, 0, -1):
        for k in range(1, K+1):
            for e in range(0, n_value):
                B[k][i][e] = B[k][i+1][e] * (1 - p[i-1]) + B[k-1][i+1][e] * p[i-1]
                if (k==1) & (value_dict[e] == f_train[i-1]):
                    B[k][i][e] += p[i-1]
    return B

*/
#[allow(non_snake_case)]
pub fn compute_boundary_set_prob_any_loss_from_tensor_predicated_simd(
    retrieved_utility_contributions: &[f64],
    distinct_utility_contributions: &[f64],
    p: &[f64],
    K: usize,
    M: usize,
    B: &mut DenseTensor,
) {
    let size_of_e = distinct_utility_contributions.len();

    // TODO maybe these should be run via SIMD as well?
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
            let matches = (distinct_utility_contributions[e] ==
                retrieved_utility_contributions[i-1]) as i64 as f64;
            B[[1,i,e]] = B[[1,i+1,e]] * (1.0 - p[i-1]) + B[[0,i+1,e]] * p[i-1] + matches * p[i-1];
        }

        for k in 2..K+1 {
            // Vectorizable version of the following code
            // for e in 0..num_labels {
            //     B[[k,i,e]] = B[[k,i+1,e]] * (1.0 - p[i-1]) + B[[k-1,i+1,e]] * p[i-1];
            // }
            B.set_y_to_x1_a1_plus_x2_a2(
                [k, i], [k, i + 1], 1.0 - p[i-1], [k - 1, i + 1], p[i - 1]
            );
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_snake_case)]
    #[test]
    fn iprp_can_reuse_matrices() {
        let M = 10;
        let K = 3;
        let p = vec![0.5; M];

        let mut IP = DenseMatrix::new(K + 1, M + 2);
        let mut RP = DenseMatrix::new(K + 1, M + 2);

        compute_prob_from_tensors(&p, K, M, &mut IP, &mut RP);

        let mut IP_clone = IP.clone();
        let mut RP_clone = RP.clone();

        compute_prob_from_tensors(&p, K, M, &mut IP_clone, &mut RP_clone);

        let norm_of_IP_difference = IP.view_buffer().iter().zip(IP_clone.view_buffer().iter())
            .take((K + 1) * (M + 2))
            .fold(0.0, |acc, (orig, clone)| acc + (orig - clone) * (orig - clone))
            .sqrt();

        assert!(norm_of_IP_difference < 0.0000001);

        let norm_of_RP_difference = RP.view_buffer().iter().zip(RP_clone.view_buffer().iter())
            .take((K + 1) * (M + 2))
            .fold(0.0, |acc, (orig, clone)| acc + (orig - clone) * (orig - clone))
            .sqrt();

        assert!(norm_of_RP_difference < 0.0000001);
    }

    // TODO add test where the shape of the reused matrix is changed
}