use ndarray::{Array1, Array2, ArrayView1, ArrayView2, azip};
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
}

impl LossFunction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BinaryCrossEntropy => "binary_cross_entropy",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "binary_cross_entropy" | "binary_crossentropy" | "binarycrossentropy" | "bce" => {
                Some(Self::BinaryCrossEntropy)
            }
            _ => None,
        }
    }

    /// Loss: L = −[y·ln p + (1−y)·ln(1−p)], p = P(positive) = last output; used for evaluation and non-softmax backprop (softmax uses delta = p − y).
    pub fn compute(&self, a: ArrayView2<'_, f64>, t: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_shape_match(a, t);
        match self {
            Self::BinaryCrossEntropy => binary_cross_entropy_loss(a, t),
        }
    }

    /// gradient: g = ∂L/∂a — BCE derivative (p−y)/(p(1−p)) on the positive (last) column; not used for softmax (delta = p − y).
    pub fn gradient(&self, a: ArrayView2<'_, f64>, t: ArrayView1<'_, f64>) -> Array2<f64> {
        assert_shape_match(a, t);
        match self {
            Self::BinaryCrossEntropy => binary_cross_entropy_gradient(a, t),
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
