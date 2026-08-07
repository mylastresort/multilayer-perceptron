use ndarray::{Array1, Array2, ArrayView1};

use crate::network::activation::ActivationFunction;
use crate::network::model::Network;
use crate::training::loss::{LossFunction, onehot_targets};

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

impl Network {
    /// backward: mean gradients (÷ n); softmax output uses delta = p − y directly, other outputs gate the loss gradient through the activation derivative.
    pub fn backward(&mut self, loss: LossFunction, t: ArrayView1<f64>) -> Vec<LayerGradients> {
        let n = t.len() as f64;
        let last = self.layers.last().expect("network has no layers");
        let p = last
            .activated_cache
            .as_ref()
            .expect("No forward pass cache; call forward first");

        let softmax_out = matches!(last.activation, ActivationFunction::Softmax);
        let mut upstream = if softmax_out {
            p - &onehot_targets(t, p.ncols())
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
