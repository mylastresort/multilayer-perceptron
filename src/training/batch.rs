use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub struct BatchIterator<'data, 'idx> {
    data: ArrayView2<'data, f64>,
    labels: ArrayView1<'data, f64>,
    indices: &'idx [usize],
    batch_size: usize,
    batch_start: usize,
}

impl<'data, 'idx> BatchIterator<'data, 'idx> {
    pub fn next_batch(&mut self) -> Option<(Array2<f64>, Array1<f64>)> {
        if self.batch_start >= self.indices.len() {
            return None;
        }

        let batch_end = (self.batch_start + self.batch_size).min(self.indices.len());
        let rows = &self.indices[self.batch_start..batch_end];

        let n = rows.len();
        let mut x_batch = Array2::zeros((n, self.data.ncols()));
        let mut y_batch = Array1::zeros(n);
        for (k, &row_idx) in rows.iter().enumerate() {
            x_batch.row_mut(k).assign(&self.data.row(row_idx));
            y_batch[k] = self.labels[row_idx];
        }

        self.batch_start = batch_end;
        Some((x_batch, y_batch))
    }

    pub fn sum_by<F>(&mut self, mut f: F) -> f64
    where
        F: FnMut(Array2<f64>, Array1<f64>) -> f64,
    {
        let mut total = 0.0;
        while let Some((x_batch, y_batch)) = self.next_batch() {
            total += f(x_batch, y_batch);
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
    }
}
