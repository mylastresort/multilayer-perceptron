use ndarray::{ArrayView1, ArrayView2, s};

pub struct BatchIterator<'data, 'idx> {
    data: ArrayView2<'data, f64>,
    labels: ArrayView1<'data, f64>,
    indices: &'idx [usize],
    batch_size: usize,
    batch_start: usize,
    in_batch_offset: usize,
}

impl<'data, 'idx> BatchIterator<'data, 'idx> {
    pub fn next<'iter>(
        &'iter mut self,
    ) -> Option<(ArrayView2<'iter, f64>, ArrayView1<'iter, f64>)> {
        if self.batch_start >= self.indices.len() {
            return None;
        }

        let batch_end = (self.batch_start + self.batch_size).min(self.indices.len());
        if self.batch_start + self.in_batch_offset >= batch_end {
            self.batch_start = batch_end;
            self.in_batch_offset = 0;
            if self.batch_start >= self.indices.len() {
                return None;
            }
        }

        let row_idx = self.indices[self.batch_start + self.in_batch_offset];
        self.in_batch_offset += 1;

        let x_row = self.data.slice(s![row_idx..row_idx + 1, ..]);
        let y_row = self.labels.slice(s![row_idx..row_idx + 1]);
        Some((x_row, y_row))
    }

    pub fn sum_by<F>(&mut self, mut f: F) -> f64
    where
        F: for<'iter> FnMut(ArrayView2<'iter, f64>, ArrayView1<'iter, f64>) -> f64,
    {
        let mut total = 0.0;
        while let Some((x_row, y_row)) = self.next() {
            total += f(x_row, y_row);
        }
        total
    }
}

pub fn create_batches<'data, 'idx>(
    data: ArrayView2<'data, f64>,
    labels: ArrayView1<'data, f64>,
    indices: &'idx [usize],
    batch_size: usize,
) -> BatchIterator<'data, 'idx> {
    BatchIterator {
        data,
        labels,
        indices,
        batch_size,
        batch_start: 0,
        in_batch_offset: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::create_batches;
    use ndarray::{Array1, Array2};

    fn data_4x2() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((4, 2), |(i, j)| (i * 2 + j) as f64);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        (x, y)
    }

    #[test]
    fn batch_iterator_visits_all_rows_once_with_batch_size_1() {
        let (x, y) = data_4x2();
        let indices: Vec<usize> = (0..4).collect();
        let mut iter = create_batches(x.view(), y.view(), &indices, 1);
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn batch_iterator_sum_by_accumulates_correctly() {
        let (x, y) = data_4x2();
        let indices: Vec<usize> = (0..4).collect();
        let mut iter = create_batches(x.view(), y.view(), &indices, 2);
        let total = iter.sum_by(|_x, _y| 1.0);
        assert_eq!(total, 4.0); // 4 rows, each contributes 1.0
    }

    #[test]
    fn batch_iterator_with_batch_size_larger_than_data() {
        let (x, y) = data_4x2();
        let indices: Vec<usize> = (0..4).collect();
        let mut iter = create_batches(x.view(), y.view(), &indices, 100);
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn batch_iterator_empty_indices_returns_none_immediately() {
        let (x, y) = data_4x2();
        let indices: Vec<usize> = vec![];
        let mut iter = create_batches(x.view(), y.view(), &indices, 2);
        assert!(iter.next().is_none());
    }

    #[test]
    fn batch_iterator_each_row_has_correct_shape() {
        let (x, y) = data_4x2();
        let indices: Vec<usize> = (0..4).collect();
        let mut iter = create_batches(x.view(), y.view(), &indices, 2);
        while let Some((x_row, y_row)) = iter.next() {
            assert_eq!(x_row.ncols(), 2);
            assert_eq!(y_row.len(), 1);
        }
    }
}
