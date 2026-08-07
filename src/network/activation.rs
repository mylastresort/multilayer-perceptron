use ndarray::{Array2, Axis, azip};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    #[serde(alias = "relu", alias = "ReLU")]
    ReLU,
    Softmax,
}

/// Activation: a = φ(z). forward computes a; backward gates an upstream gradient through φ'(a) for sigmoid/tanh/ReLU; softmax is output-only (delta = p − y).
pub trait Activation {
    /// forward: a = φ(z).
    fn forward(&self, z: &Array2<f64>) -> Array2<f64>;
    /// backward: ∂L/∂z = g ⊙ φ'(a) for element-wise φ; panics for softmax (use delta = p − y).
    fn backward(&self, a: &Array2<f64>, g: &Array2<f64>) -> Array2<f64>;
}

impl Activation for ActivationFunction {
    /// forward: a = φ(z).
    #[inline]
    fn forward(&self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFunction::Sigmoid => z.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => z.mapv(f64::tanh),
            ActivationFunction::ReLU => z.mapv(|v| v.max(0.0)),
            ActivationFunction::Softmax => {
                let mut result = Array2::<f64>::zeros(z.raw_dim());

                azip!((in_row in z.axis_iter(Axis(0)), mut out_row in result.axis_iter_mut(Axis(0))) {
                    let max = in_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    // softmax(z_i) = e^(z_i − M) / Σ_k e^(z_k − M)
                    azip!((o in &mut out_row, &v in in_row) { *o = (v - max).exp(); });
                    let sum = out_row.sum();
                    azip!((o in &mut out_row) { *o /= sum; });
                });

                result
            }
        }
    }

    /// backward: ∂L/∂z = g ⊙ φ'(a) for element-wise φ.
    #[inline]
    fn backward(&self, a: &Array2<f64>, g: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFunction::Sigmoid => {
                let deriv = a.mapv(|v| v * (1.0 - v));
                g * deriv
            }
            ActivationFunction::Tanh => {
                let deriv = a.mapv(|v| 1.0 - v * v);
                g * deriv
            }
            ActivationFunction::ReLU => {
                let deriv = a.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                g * deriv
            }
            ActivationFunction::Softmax => {
                panic!(
                    "softmax has no element-wise backward: it is output-only and its delta is p − y"
                )
            }
        }
    }
}
