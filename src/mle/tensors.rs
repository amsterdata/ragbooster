use std::ops::{Index, IndexMut};
use std::slice;

#[derive(Clone)]
pub struct DenseMatrix {
    num_rows: usize,
    num_columns: usize,
    buffer: Vec<f64>,
}

impl DenseMatrix {
    pub fn new(num_rows: usize, num_columns: usize) -> Self {
        Self::allocate_with_capacity(num_rows, num_columns, num_rows * num_columns)
    }

    pub fn allocate_with_capacity(num_rows: usize, num_columns: usize, capacity: usize) -> Self {
        assert!(capacity >= num_rows * num_columns);
        let buffer = vec![0_f64; capacity];
        Self { num_rows, num_columns, buffer }
    }

    // Note: this does NOT reset the underlying buffer to zeros!
    pub fn reuse_as(&mut self, num_rows: usize, num_columns: usize) {
        assert!(self.buffer.len() >= num_rows * num_columns);
        self.num_rows = num_rows;
        self.num_columns = num_columns;
    }

    #[allow(unused)] // used for testing only
    pub fn view_buffer(&self) -> &[f64] {
        &self.buffer
    }
}

impl Index<[usize; 2]> for DenseMatrix {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        unsafe {
            let row = index.get_unchecked(0);
            let column = index.get_unchecked(1);
            let position = row * self.num_columns + column;
            self.buffer.get_unchecked(position)
        }
    }
}

impl IndexMut<[usize; 2]> for DenseMatrix {
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        unsafe {
            let row = index.get_unchecked(0);
            let column = index.get_unchecked(1);
            let position = row * self.num_columns + column;
            self.buffer.get_unchecked_mut(position)
        }
    }
}

pub struct DenseTensor {
    #[allow(unused)]
    dim_1: usize,
    dim_2: usize,
    dim_3: usize,
    buffer: Vec<f64>,
}

const CHUNK_SIZE: usize = 4;

impl DenseTensor {

    #[allow(non_snake_case)]
    pub fn new(dim_1: usize, dim_2: usize, dim_3: usize) -> Self {
        let buffer = vec![0_f64; dim_1 * dim_2 * dim_3];
        Self { dim_1, dim_2, dim_3, buffer }
    }

    // Note: this does NOT reset the underlying buffer to zeros!
    pub fn reuse_as(&mut self, dim_1: usize, dim_2: usize, dim_3: usize) {
        assert!(self.buffer.len() >= dim_1 * dim_2 * dim_3);
        self.dim_1 = dim_1;
        self.dim_2 = dim_2;
        self.dim_3 = dim_3;
    }

    #[inline(always)]
    pub(crate) fn set_y_to_x1_a1_plus_x2_a2(
        &mut self,
        y_indices: [usize; 2],
        x1_indices: [usize; 2],
        a1: f64,
        x2_indices: [usize; 2],
        a2: f64,
    ) {

        let y_offset =
            (y_indices[0] * self.dim_2 * self.dim_3 + y_indices[1] * self.dim_3) as isize;
        let x1_offset =
            (x1_indices[0] * self.dim_2 * self.dim_3 + x1_indices[1] * self.dim_3) as isize;
        let x2_offset =
            (x2_indices[0] * self.dim_2 * self.dim_3 + x2_indices[1] * self.dim_3) as isize;

        let y_vec = unsafe {
            slice::from_raw_parts_mut(self.buffer.as_mut_ptr().offset(y_offset), self.dim_3)
        };
        let x1_vec = unsafe {
            slice::from_raw_parts(self.buffer.as_ptr().offset(x1_offset), self.dim_3)
        };
        let x2_vec = unsafe {
            slice::from_raw_parts(self.buffer.as_ptr().offset(x2_offset), self.dim_3)
        };

        let y_chunked = y_vec.chunks_exact_mut(CHUNK_SIZE);
        let x1_chunked = x1_vec.chunks_exact(CHUNK_SIZE);
        let x2_chunked = x2_vec.chunks_exact(CHUNK_SIZE);

        let outer_iter = y_chunked.zip(x1_chunked).zip(x2_chunked);

        for ((y_chunk, x1_chunk), x2_chunk) in outer_iter {
            let inner_iter = y_chunk.iter_mut().zip(x1_chunk.iter()).zip(x2_chunk.iter());
            for ((y, x1), x2) in inner_iter {
                *y = *x1 * a1 + *x2 * a2;
            }
        }

        let y_remainder = y_vec.chunks_exact_mut(CHUNK_SIZE).into_remainder().iter_mut();
        let x1_remainder = x1_vec.chunks_exact(CHUNK_SIZE).remainder().iter();
        let x2_remainder = x2_vec.chunks_exact(CHUNK_SIZE).remainder().iter();

        let remainders = y_remainder.zip(x1_remainder).zip(x2_remainder);

        for ((y, x1), x2) in remainders {
            *y = *x1 * a1 + *x2 * a2;
        }
    }

    #[allow(unused)] // used for testing only
    pub fn view_buffer(&self) -> &[f64] {
        &self.buffer
    }
}

impl Index<[usize; 3]> for DenseTensor {
    type Output = f64;

    #[inline(always)]
    fn index(&self, index: [usize; 3]) -> &Self::Output {
        unsafe {
            let index_1 = index.get_unchecked(0);
            let index_2 = index.get_unchecked(1);
            let index_3 = index.get_unchecked(2);
            let position = index_1 * self.dim_2 * self.dim_3 + index_2 * self.dim_3 + index_3;
            self.buffer.get_unchecked(position)
        }
    }
}

impl IndexMut<[usize; 3]> for DenseTensor {
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        unsafe {
            let index_1 = index.get_unchecked(0);
            let index_2 = index.get_unchecked(1);
            let index_3 = index.get_unchecked(2);
            let position = index_1 * self.dim_2 * self.dim_3 + index_2 * self.dim_3 + index_3;
            self.buffer.get_unchecked_mut(position)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn basic() {
        let mut m = DenseMatrix::new(50, 100);

        assert_eq!(0.0, m[[0,0]]);
        assert_eq!(0.0, m[[49,99]]);

        m[[0,5]] = 3.0;
        assert_eq!(3.0, m[[0,5]]);
    }

    #[test]
    fn with_reuse() {
        let mut m = DenseMatrix::allocate_with_capacity(10, 100, 10000);

        m[[0, 0]] = 5.0;
        assert_eq!(5.0, m[[0, 0]]);

        m.reuse_as(50, 200);

        m[[49,199]] = 3.0;
        assert_eq!(3.0, m[[49,199]]);
    }

    #[test]
    fn set_y_to_x1_a1_plus_x2_a2() {

        let mut m = DenseTensor::new(20, 10, 5);

        m[[0,0,0]] = 1.0;
        m[[0,0,1]] = 2.0;
        m[[0,0,2]] = 3.0;
        m[[0,0,3]] = 4.0;
        m[[0,0,4]] = 5.0;

        m[[3,2,0]] = 1.0;
        m[[3,2,1]] = 2.0;
        m[[3,2,2]] = 3.0;
        m[[3,2,3]] = 4.0;
        m[[3,2,4]] = 5.0;

        m[[19,9,0]] = 1.0;
        m[[19,9,1]] = 2.0;
        m[[19,9,2]] = 3.0;
        m[[19,9,3]] = 4.0;
        m[[19,9,4]] = 5.0;

        m.set_y_to_x1_a1_plus_x2_a2([0,0], [3,2], 0.1, [19,9], 0.2);

        let epsilon = 0.00000001;

        assert_abs_diff_eq!(m[[0,0,0]], 0.3, epsilon=epsilon);
        assert_abs_diff_eq!(m[[0,0,1]], 0.6, epsilon=epsilon);
        assert_abs_diff_eq!(m[[0,0,2]], 0.9, epsilon=epsilon);
        assert_abs_diff_eq!(m[[0,0,3]], 1.2, epsilon=epsilon);
        assert_abs_diff_eq!(m[[0,0,4]], 1.5, epsilon=epsilon);
    }
}