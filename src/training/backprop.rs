use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::network::{activation::ActivationFunction, model::Network};
use crate::training::loss::LossFunction;

#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl From<(Array2<f64>, Array1<f64>)> for LayerGradients {
    fn from((weights, bias): (Array2<f64>, Array1<f64>)) -> Self {
        Self { weights, bias }
    }
}

impl std::ops::DivAssign<f64> for LayerGradients {
    fn div_assign(&mut self, n: f64) {
        self.weights /= n;
        self.bias /= n;
    }
}

/// y for delta = p − y: column vector (ncols == 1) or one-hot (targets 0/1).
fn onehot_encoded(t: ArrayView1<f64>, ncols: usize) -> Array2<f64> {
    if ncols == 1 {
        t.to_owned().insert_axis(Axis(1))
    } else {
        let mut y = Array2::zeros((t.len(), ncols));
        for (mut row, &target) in y.outer_iter_mut().zip(t) {
            row[binary_class_index(target)] = 1.0;
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

impl Network {
    /// backward: mean gradients (÷ n); softmax output uses delta = p − y directly, other outputs gate the BCE gradient through the activation derivative.
    pub fn backward(&mut self, loss: LossFunction, t: ArrayView1<f64>) -> Vec<LayerGradients> {
        let n = t.len() as f64;
        let last = self.layers.last().expect("network has no layers");
        let p = last
            .activated_cache
            .as_ref()
            .expect("No forward pass cache; call forward first");

        let softmax_out = matches!(last.activation, ActivationFunction::Softmax);
        let mut upstream = if softmax_out {
            p - &onehot_encoded(t, p.ncols())
        } else {
            loss.gradient(p.view(), t) / n
        };

        let mut gd = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().rev().enumerate() {
            let (g_input, g_weights, g_bias) = if i == 0 && softmax_out {
                layer.backward_with_delta(&upstream)
            } else {
                layer.backward(&upstream)
            };
            gd.push(LayerGradients::from((g_weights, g_bias)));
            upstream = g_input;
        }

        gd.reverse();
        if softmax_out {
            for g in gd.iter_mut() {
                *g /= n;
            }
        }
        gd
    }
}
