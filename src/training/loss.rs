use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, azip};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossFunction {
    #[default]
    #[serde(
        alias = "binary_crossentropy",
        alias = "binaryCrossentropy",
        alias = "bce"
    )]
    BinaryCrossEntropy,
    #[serde(
        alias = "categorical_crossentropy",
        alias = "categoricalCrossentropy",
        alias = "categoricalcrossentropy",
        alias = "cce"
    )]
    CategoricalCrossEntropy,
}

impl LossFunction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BinaryCrossEntropy => "binary_cross_entropy",
            Self::CategoricalCrossEntropy => "categorical_cross_entropy",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
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

    /// Loss: L = −[y·ln p + (1−y)·ln(1−p)], p = P(positive) = last output; used for evaluation and non-softmax backprop (softmax uses delta = p − y).
    pub fn compute(&self, a: ArrayView2<'_, f64>, t: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_shape_match(a, t);
        match self {
            Self::BinaryCrossEntropy => binary_cross_entropy_loss(a, t),
            Self::CategoricalCrossEntropy => categorical_cross_entropy_loss(a, t),
        }
    }

    /// gradient: g = ∂L/∂a — BCE derivative (p−y)/(p(1−p)) on the positive (last) column; not used for softmax (delta = p − y).
    pub fn gradient(&self, a: ArrayView2<'_, f64>, t: ArrayView1<'_, f64>) -> Array2<f64> {
        assert_shape_match(a, t);
        match self {
            Self::BinaryCrossEntropy => binary_cross_entropy_gradient(a, t),
            Self::CategoricalCrossEntropy => categorical_cross_entropy_gradient(a, t),
        }
    }
}

#[inline]
fn assert_shape_match(predictions: ArrayView2<'_, f64>, targets: ArrayView1<'_, f64>) {
    if predictions.nrows() != targets.len() {
        panic!(
            "predictions rows ({}) must match targets len ({})",
            predictions.nrows(),
            targets.len()
        );
    }
}

/// Assert the binary output width (1 or 2) before training/prediction.
pub fn assert_binary_output(output_width: usize) {
    assert!(
        output_width <= 2,
        "binary cross entropy supports a 1- or 2-output network, got {output_width} outputs"
    );
}

const EPS: f64 = 1e-12;

fn binary_cross_entropy_loss(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array1<f64> {
    // subject formula: L = −[y ln p + (1−y) ln(1−p)], p = positive class (last output).
    let p_col = predictions.ncols() - 1;
    let mut losses = Array1::zeros(predictions.nrows());
    azip!((&p in predictions.column(p_col), &y in targets, ls in &mut losses) {
        let p = p.clamp(EPS, 1.0 - EPS);
        let y = y.clamp(0.0, 1.0);
        *ls = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    });
    losses
}

fn binary_cross_entropy_gradient(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array2<f64> {
    // g = ∂L/∂p = (p − y)/(p(1−p)) on the positive-class (last) output, 0 elsewhere.
    let p_col = predictions.ncols() - 1;
    let mut grad = Array2::zeros((predictions.nrows(), predictions.ncols()));
    azip!((&p in predictions.column(p_col), &y in targets, g in grad.column_mut(p_col)) {
        let p = p.clamp(EPS, 1.0 - EPS);
        let y = y.clamp(0.0, 1.0);
        *g = (p - y) / (p * (1.0 - p));
    });
    grad
}

/// One-hot encode single-label binary targets (class indices 0/1); a single-output
/// network returns the target column unchanged (ncols == 1).
pub(crate) fn onehot_targets(targets: ArrayView1<'_, f64>, ncols: usize) -> Array2<f64> {
    if ncols == 1 {
        targets.to_owned().insert_axis(Axis(1))
    } else {
        let mut y = Array2::zeros((targets.len(), ncols));
        for (mut row, &label) in y.outer_iter_mut().zip(targets) {
            let k = binary_class_index(label);
            assert!(
                k < ncols,
                "categorical target {label} is out of range for a {ncols}-output network"
            );
            row[k] = 1.0;
        }
        y
    }
}

fn binary_class_index(target: f64) -> usize {
    match target {
        0.0 => 0,
        1.0 => 1,
        other => panic!("invalid binary target {other}: BCE expects class index 0 or 1"),
    }
}

/// Categorical cross-entropy: L = −Σ_k y_k·ln p_k with one-hot targets y; p_k clamped to avoid log(0).
fn categorical_cross_entropy_loss(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let y = onehot_targets(targets, predictions.ncols());
    let mut losses = Array1::zeros(predictions.nrows());
    for ((row, y_row), ls) in predictions
        .outer_iter()
        .zip(y.outer_iter())
        .zip(losses.iter_mut())
    {
        let mut sum = 0.0;
        for (p, &t) in row.iter().zip(y_row.iter()) {
            let p = p.clamp(EPS, 1.0 - EPS);
            sum += t * p.ln();
        }
        *ls = -sum;
    }
    losses
}

/// Categorical cross-entropy gradient: g = ∂L/∂a_k = −y_k/p_k elementwise; only used for non-softmax backprop (softmax uses delta = p − y).
fn categorical_cross_entropy_gradient(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let y = onehot_targets(targets, predictions.ncols());
    let mut grad = Array2::zeros(predictions.raw_dim());
    azip!((g in &mut grad, &p in &predictions, &t in &y) {
        let p = p.clamp(EPS, 1.0 - EPS);
        *g = -(t / p);
    });
    grad
}
