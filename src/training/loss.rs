use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

/// Supported loss functions for training.
///
/// `CategoricalCrossEntropy` is the default: for the 2-class breast-cancer
/// problem it is equivalent to binary cross-entropy (BCE with one-hot targets),
/// which is why training and prediction share the same loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossFunction {
    MSE,
    #[serde(
        alias = "binary_crossentropy",
        alias = "binaryCrossEntropy",
        alias = "bce"
    )]
    BinaryCrossEntropy,
    #[default]
    #[serde(
        alias = "categorical_crossentropy",
        alias = "categoricalCrossentropy",
        alias = "cce"
    )]
    CategoricalCrossEntropy,
}

impl LossFunction {
    /// Human-readable lowercase identifier, used for CLI errors and display.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MSE => "mse",
            Self::BinaryCrossEntropy => "binary_cross_entropy",
            Self::CategoricalCrossEntropy => "categorical_cross_entropy",
        }
    }

    /// Parses a loss name from the CLI, accepting the spellings used by the
    /// subject and Keras-style conventions.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mse" => Some(Self::MSE),
            "binary_cross_entropy" | "binary_crossentropy" | "binarycrossentropy" | "bce" => {
                Some(Self::BinaryCrossEntropy)
            }
            "categorical_cross_entropy"
            | "categorical_crossentropy"
            | "categoricalcrossentropy"
            | "cce" => Some(Self::CategoricalCrossEntropy),
            _ => None,
        }
    }
}

/// Trait for loss functions, providing per-sample loss and gradient computation.
///
/// Neither `compute` nor `gradient` reduce over the batch: both return one
/// value per sample. Reducing to a scalar (mean or sum) is a training concern
/// and is handled by the caller (e.g. the trainer).
pub trait Loss {
    /// Computes the per-sample loss for each sample in the batch.
    fn compute(
        &self,
        predictions: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
    ) -> Array1<f64>;
    /// Computes the per-sample gradient of the loss with respect to predictions.
    fn gradient(
        &self,
        predictions: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
    ) -> Array2<f64>;
}

fn assert_shape_match(predictions: ArrayView2<'_, f64>, targets: ArrayView1<'_, f64>) {
    if predictions.nrows() != targets.len() {
        panic!(
            "predictions rows ({}) must match targets len ({})",
            predictions.nrows(),
            targets.len()
        );
    }
}

fn class_index(target: f64, n_classes: usize) -> usize {
    target
        .round()
        .clamp(0.0, (n_classes - 1) as f64) as usize
}

fn mse_loss(predictions: ArrayView2<'_, f64>, targets: ArrayView1<'_, f64>) -> Array1<f64> {
    let cols = predictions.ncols();
    let mut losses = Array1::zeros(predictions.nrows());
    for (row_idx, row) in predictions.outer_iter().enumerate() {
        let target = targets[row_idx];
        let mut total = 0.0;
        for col_idx in 0..cols {
            let diff = row[col_idx] - target;
            total += diff * diff;
        }
        losses[row_idx] = total / (cols as f64);
    }
    losses
}

fn binary_cross_entropy_loss(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let eps = 1e-12;
    let mut losses = Array1::zeros(predictions.nrows());
    if predictions.ncols() == 1 {
        for (row_idx, row) in predictions.outer_iter().enumerate() {
            let y = targets[row_idx].clamp(0.0, 1.0);
            let p = row[0].clamp(eps, 1.0 - eps);
            losses[row_idx] = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        }
    } else {
        for (row_idx, row) in predictions.outer_iter().enumerate() {
            let p = row[class_index(targets[row_idx], predictions.ncols())].clamp(eps, 1.0 - eps);
            losses[row_idx] = -p.ln();
        }
    }
    losses
}

fn categorical_cross_entropy_loss(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let eps = 1e-12;
    let mut losses = Array1::zeros(predictions.nrows());
    for (row_idx, row) in predictions.outer_iter().enumerate() {
        let p = row[class_index(targets[row_idx], predictions.ncols())].clamp(eps, 1.0 - eps);
        losses[row_idx] = -p.ln();
    }
    losses
}

fn mse_gradient(predictions: ArrayView2<'_, f64>, targets: ArrayView1<'_, f64>) -> Array2<f64> {
    let scale = 2.0 / (predictions.ncols() as f64);
    let mut grad = predictions.to_owned();
    for (row_idx, mut row) in grad.outer_iter_mut().enumerate() {
        let target = targets[row_idx];
        for col_idx in 0..row.len() {
            row[col_idx] = (row[col_idx] - target) * scale;
        }
    }
    grad
}

fn binary_cross_entropy_gradient(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let eps = 1e-12;
    if predictions.ncols() == 1 {
        let mut grad = Array2::zeros((predictions.nrows(), 1));
        for (row_idx, row) in predictions.outer_iter().enumerate() {
            let y = targets[row_idx].clamp(0.0, 1.0);
            let p = row[0].clamp(eps, 1.0 - eps);
            grad[[row_idx, 0]] = (p - y) / (p * (1.0 - p));
        }
        grad
    } else {
        let mut grad = predictions.to_owned();
        for (row_idx, mut row) in grad.outer_iter_mut().enumerate() {
            let class_idx = class_index(targets[row_idx], predictions.ncols());
            row[class_idx] -= 1.0;
        }
        grad
    }
}

fn categorical_cross_entropy_gradient(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let mut grad = predictions.to_owned();
    for (row_idx, mut row) in grad.outer_iter_mut().enumerate() {
        let class_idx = class_index(targets[row_idx], predictions.ncols());
        row[class_idx] -= 1.0;
    }
    grad
}

impl Loss for LossFunction {
    fn compute(
        &self,
        predictions: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_shape_match(predictions, targets);
        match self {
            LossFunction::MSE => mse_loss(predictions, targets),
            LossFunction::BinaryCrossEntropy => binary_cross_entropy_loss(predictions, targets),
            LossFunction::CategoricalCrossEntropy => {
                categorical_cross_entropy_loss(predictions, targets)
            }
        }
    }

    fn gradient(
        &self,
        predictions: ArrayView2<'_, f64>,
        targets: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        assert_shape_match(predictions, targets);
        match self {
            LossFunction::MSE => mse_gradient(predictions, targets),
            LossFunction::BinaryCrossEntropy => {
                binary_cross_entropy_gradient(predictions, targets)
            }
            LossFunction::CategoricalCrossEntropy => {
                categorical_cross_entropy_gradient(predictions, targets)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Loss, LossFunction};
    use ndarray::{arr1, arr2};

    // -----------------------------------------------------------------------
    // LossFunction naming / parsing
    // -----------------------------------------------------------------------

    #[test]
    fn loss_function_parse_accepts_common_spellings() {
        assert_eq!(LossFunction::parse("mse"), Some(LossFunction::MSE));
        assert_eq!(
            LossFunction::parse("binary_cross_entropy"),
            Some(LossFunction::BinaryCrossEntropy)
        );
        assert_eq!(
            LossFunction::parse("binary_crossentropy"),
            Some(LossFunction::BinaryCrossEntropy)
        );
        assert_eq!(
            LossFunction::parse("categorical_cross_entropy"),
            Some(LossFunction::CategoricalCrossEntropy)
        );
        assert_eq!(
            LossFunction::parse("categoricalCrossentropy"),
            Some(LossFunction::CategoricalCrossEntropy)
        );
        assert_eq!(
            LossFunction::parse("cce"),
            Some(LossFunction::CategoricalCrossEntropy)
        );
        assert_eq!(LossFunction::parse("bogus"), None);
    }

    #[test]
    fn loss_function_serde_round_trips() {
        for loss in [
            LossFunction::MSE,
            LossFunction::BinaryCrossEntropy,
            LossFunction::CategoricalCrossEntropy,
        ] {
            let json = serde_json::to_string(&loss).unwrap();
            let back: LossFunction = serde_json::from_str(&json).unwrap();
            assert_eq!(loss, back, "serde round-trip failed for {json}");
        }
        // Default is categorical cross-entropy (the 2-class BCE-equivalent).
        assert_eq!(
            LossFunction::default(),
            LossFunction::CategoricalCrossEntropy
        );
    }

    // -----------------------------------------------------------------------
    // MSE
    // -----------------------------------------------------------------------

    #[test]
    fn mse_compute_zero_when_predictions_equal_targets() {
        let preds = arr2(&[[1.0], [2.0], [3.0]]);
        let targets = arr1(&[1.0, 2.0, 3.0]);
        let losses = LossFunction::MSE.compute(preds.view(), targets.view());
        for v in losses.iter() {
            assert!(v.abs() < 1e-12, "expected zero loss, got {v}");
        }
    }

    #[test]
    fn mse_compute_correct_value() {
        // predictions = [[2.0]], targets = [0.0]  → per-sample MSE = (2-0)^2 / 1 = 4.0
        let preds = arr2(&[[2.0]]);
        let targets = arr1(&[0.0]);
        let losses = LossFunction::MSE.compute(preds.view(), targets.view());
        assert_eq!(losses.len(), 1);
        assert!((losses[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn mse_compute_returns_one_loss_per_sample() {
        let preds = arr2(&[[1.0], [2.0], [3.0]]);
        let targets = arr1(&[1.0, 2.0, 3.0]);
        let losses = LossFunction::MSE.compute(preds.view(), targets.view());
        assert_eq!(losses.len(), preds.nrows());
    }

    #[test]
    fn mse_gradient_zero_when_predictions_equal_targets() {
        let preds = arr2(&[[1.0], [2.0]]);
        let targets = arr1(&[1.0, 2.0]);
        let grad = LossFunction::MSE.gradient(preds.view(), targets.view());
        for v in grad.iter() {
            assert!(v.abs() < 1e-12, "expected zero gradient, got {v}");
        }
    }

    #[test]
    fn mse_gradient_correct_sign() {
        // pred > target → gradient should be positive
        let preds = arr2(&[[3.0]]);
        let targets = arr1(&[1.0]);
        let grad = LossFunction::MSE.gradient(preds.view(), targets.view());
        assert!(grad[[0, 0]] > 0.0);
    }

    #[test]
    #[should_panic]
    fn mse_compute_panics_on_row_mismatch() {
        let preds = arr2(&[[1.0], [2.0]]);
        let targets = arr1(&[1.0]);
        LossFunction::MSE.compute(preds.view(), targets.view());
    }

    // -----------------------------------------------------------------------
    // BinaryCrossEntropy – single column
    // -----------------------------------------------------------------------

    #[test]
    fn bce_single_col_compute_near_zero_for_perfect_predictions() {
        // p ≈ 1 for y=1 → loss ≈ 0
        let preds = arr2(&[[0.999_999]]);
        let targets = arr1(&[1.0]);
        let losses = LossFunction::BinaryCrossEntropy.compute(preds.view(), targets.view());
        assert!(
            losses[0] < 1e-4,
            "expected near-zero loss, got {}",
            losses[0]
        );
    }

    #[test]
    fn bce_single_col_gradient_direction_for_overestimate() {
        // p > y → gradient should be positive
        let preds = arr2(&[[0.9]]);
        let targets = arr1(&[0.0]);
        let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
        assert!(grad[[0, 0]] > 0.0);
    }

    #[test]
    fn bce_single_col_gradient_direction_for_underestimate() {
        // p < y → gradient should be negative
        let preds = arr2(&[[0.1]]);
        let targets = arr1(&[1.0]);
        let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
        assert!(grad[[0, 0]] < 0.0);
    }

    // -----------------------------------------------------------------------
    // BinaryCrossEntropy – multi column (one-hot style)
    // -----------------------------------------------------------------------

    #[test]
    fn bce_multi_col_compute_finite_for_valid_predictions() {
        let preds = arr2(&[[0.8, 0.2], [0.3, 0.7]]);
        let targets = arr1(&[0.0, 1.0]);
        let losses = LossFunction::BinaryCrossEntropy.compute(preds.view(), targets.view());
        assert!(losses.iter().all(|l| l.is_finite()));
        assert!(losses.iter().all(|l| *l > 0.0));
    }

    #[test]
    fn bce_multi_col_gradient_shape_matches_predictions() {
        let preds = arr2(&[[0.6, 0.4], [0.3, 0.7]]);
        let targets = arr1(&[0.0, 1.0]);
        let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
        assert_eq!(grad.dim(), preds.dim());
    }

    // -----------------------------------------------------------------------
    // CategoricalCrossEntropy
    // -----------------------------------------------------------------------

    #[test]
    fn cce_compute_finite_for_valid_predictions() {
        let preds = arr2(&[[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]);
        let targets = arr1(&[1.0, 0.0]);
        let losses = LossFunction::CategoricalCrossEntropy.compute(preds.view(), targets.view());
        assert!(losses.iter().all(|l| l.is_finite()));
        assert!(losses.iter().all(|l| *l > 0.0));
    }

    #[test]
    fn cce_gradient_shape_matches_predictions() {
        let preds = arr2(&[[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]);
        let targets = arr1(&[1.0, 0.0]);
        let grad = LossFunction::CategoricalCrossEntropy.gradient(preds.view(), targets.view());
        assert_eq!(grad.dim(), preds.dim());
    }

    #[test]
    fn cce_gradient_subtracts_one_from_true_class() {
        // Single sample; class 1 is true → grad[0][1] should be reduced
        let preds = arr2(&[[0.2, 0.5, 0.3]]);
        let targets = arr1(&[1.0]);
        let grad = LossFunction::CategoricalCrossEntropy.gradient(preds.view(), targets.view());
        // grad[0][1] = 0.5 - 1.0 = -0.5
        assert!((grad[[0, 1]] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn cce_gradient_panics_on_row_mismatch() {
        let preds = arr2(&[[0.5, 0.5]]);
        let targets = arr1(&[0.0, 1.0]);
        LossFunction::CategoricalCrossEntropy.gradient(preds.view(), targets.view());
    }
}
