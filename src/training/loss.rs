use ndarray::{Array2, ArrayView1, ArrayView2};

pub enum LossFunction {
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
}

pub trait Loss {
    fn compute(&self, predictions: ArrayView2<'_, f64>, targets: ArrayView1<'_, f64>) -> f64;
    fn gradient(
        &self,
        predictions: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
    ) -> Array2<f64>;
}

impl Loss for LossFunction {
    fn compute(&self, predictions: ArrayView2<'_, f64>, targets: ArrayView1<'_, f64>) -> f64 {
        if predictions.nrows() != targets.len() {
            panic!(
                "predictions rows ({}) must match targets len ({})",
                predictions.nrows(),
                targets.len()
            );
        }

        match self {
            LossFunction::MSE => {
                let cols = predictions.ncols();
                let mut total = 0.0;
                for (row_idx, row) in predictions.outer_iter().enumerate() {
                    let target = targets[row_idx];
                    for col_idx in 0..cols {
                        let diff = row[col_idx] - target;
                        total += diff * diff;
                    }
                }
                total / ((predictions.nrows() * cols) as f64)
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = 1e-12;
                if predictions.ncols() == 1 {
                    let mut total = 0.0;
                    for (row_idx, row) in predictions.outer_iter().enumerate() {
                        let y = targets[row_idx].clamp(0.0, 1.0);
                        let p = row[0].clamp(eps, 1.0 - eps);
                        total += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
                    }
                    total / (predictions.nrows() as f64)
                } else {
                    let mut total = 0.0;
                    for (row_idx, row) in predictions.outer_iter().enumerate() {
                        let class_idx = targets[row_idx]
                            .round()
                            .clamp(0.0, (predictions.ncols() - 1) as f64)
                            as usize;
                        let p = row[class_idx].clamp(eps, 1.0 - eps);
                        total += -p.ln();
                    }
                    total / (predictions.nrows() as f64)
                }
            }
            LossFunction::CategoricalCrossEntropy => {
                let eps = 1e-12;
                let mut total = 0.0;
                for (row_idx, row) in predictions.outer_iter().enumerate() {
                    let class_idx = targets[row_idx]
                        .round()
                        .clamp(0.0, (predictions.ncols() - 1) as f64)
                        as usize;
                    let p = row[class_idx].clamp(eps, 1.0 - eps);
                    total += -p.ln();
                }
                total / (predictions.nrows() as f64)
            }
        }
    }

    fn gradient(
        &self,
        predictions: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        if predictions.nrows() != targets.len() {
            panic!(
                "predictions rows ({}) must match targets len ({})",
                predictions.nrows(),
                targets.len()
            );
        }

        match self {
            LossFunction::MSE => {
                let scale = 2.0 / ((predictions.nrows() * predictions.ncols()) as f64);
                let mut grad = predictions.to_owned();
                for (row_idx, mut row) in grad.outer_iter_mut().enumerate() {
                    let target = targets[row_idx];
                    for col_idx in 0..row.len() {
                        row[col_idx] = (row[col_idx] - target) * scale;
                    }
                }
                grad
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = 1e-12;
                if predictions.ncols() == 1 {
                    let mut grad = Array2::zeros((predictions.nrows(), 1));
                    let scale = 1.0 / (predictions.nrows() as f64);
                    for (row_idx, row) in predictions.outer_iter().enumerate() {
                        let y = targets[row_idx].clamp(0.0, 1.0);
                        let p = row[0].clamp(eps, 1.0 - eps);
                        grad[[row_idx, 0]] = ((p - y) / (p * (1.0 - p))) * scale;
                    }
                    grad
                } else {
                    let mut grad = predictions.to_owned();
                    let scale = 1.0 / (predictions.nrows() as f64);
                    for (row_idx, mut row) in grad.outer_iter_mut().enumerate() {
                        let class_idx = targets[row_idx]
                            .round()
                            .clamp(0.0, (predictions.ncols() - 1) as f64)
                            as usize;
                        row[class_idx] -= 1.0;
                        for col_idx in 0..row.len() {
                            row[col_idx] *= scale;
                        }
                    }
                    grad
                }
            }
            LossFunction::CategoricalCrossEntropy => {
                let mut grad = predictions.to_owned();
                let scale = 1.0 / (predictions.nrows() as f64);
                for (row_idx, mut row) in grad.outer_iter_mut().enumerate() {
                    let class_idx = targets[row_idx]
                        .round()
                        .clamp(0.0, (predictions.ncols() - 1) as f64)
                        as usize;
                    row[class_idx] -= 1.0;
                    for col_idx in 0..row.len() {
                        row[col_idx] *= scale;
                    }
                }
                grad
            }
        }
    }
}
