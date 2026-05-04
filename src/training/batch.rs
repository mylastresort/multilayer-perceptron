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
