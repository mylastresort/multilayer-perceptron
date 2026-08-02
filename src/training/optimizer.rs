use serde::Deserialize;

use crate::{network::model::Network, training::backprop::LayerGradients};

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Sgd,
    Adam,
}

pub enum OptimizerType {
    SGD,
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    },
}

pub trait Optimizer {
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]);
    fn set_lr(&mut self, lr: f64);
}

pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]) {
        assert_eq!(
            network.layers.len(),
            gradients.len(),
            "network layer count and gradient count must match"
        );
        for (layer, grad) in network.layers.iter_mut().zip(gradients.iter()) {
            layer.weights = &layer.weights - &(grad.weights.clone() * self.learning_rate);
            layer.bias = &layer.bias - &(grad.bias.clone() * self.learning_rate);
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    t: usize,
    m: Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
    v: Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    fn ensure_moments(&mut self, network: &Network) {
        if self.m.is_empty() {
            self.m = network
                .layers
                .iter()
                .map(|l| {
                    (
                        ndarray::Array2::zeros(l.weights.raw_dim()),
                        ndarray::Array1::zeros(l.bias.raw_dim()),
                    )
                })
                .collect();
            self.v = self.m.clone();
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]) {
        assert_eq!(network.layers.len(), gradients.len());
        self.ensure_moments(network);

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let decay = 1.0 - self.learning_rate * self.weight_decay;

        for ((layer, grad), ((mw, mb), (vw, vb))) in network
            .layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            *mw = &(*mw) * self.beta1 + &(grad.weights.clone() * (1.0 - self.beta1));
            *mb = &(*mb) * self.beta1 + &(grad.bias.clone() * (1.0 - self.beta1));

            *vw = &(*vw) * self.beta2 + &(grad.weights.mapv(|v| v * v) * (1.0 - self.beta2));
            *vb = &(*vb) * self.beta2 + &(grad.bias.mapv(|v| v * v) * (1.0 - self.beta2));

            let mw_hat = mw.mapv(|v| v / bc1);
            let mb_hat = mb.mapv(|v| v / bc1);
            let vw_hat = vw.mapv(|v| v / bc2);
            let vb_hat = vb.mapv(|v| v / bc2);

            layer.weights = &layer.weights * decay
                - &(mw_hat * self.learning_rate / vw_hat.mapv(|v| v.sqrt() + self.epsilon));
            layer.bias = &layer.bias
                - &(mb_hat * self.learning_rate / vb_hat.mapv(|v| v.sqrt() + self.epsilon));
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

impl OptimizerType {
    pub fn create(&self, learning_rate: f64) -> Box<dyn Optimizer> {
        match self {
            OptimizerType::SGD => Box::new(SGD::new(learning_rate)),
            OptimizerType::Adam {
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => Box::new(
                Adam::new(learning_rate, *beta1, *beta2, *epsilon).weight_decay(*weight_decay),
            ),
        }
    }

    pub fn for_kind(kind: OptimizerKind, weight_decay: f64) -> Self {
        match kind {
            OptimizerKind::Sgd => OptimizerType::SGD,
            OptimizerKind::Adam => OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay,
            },
        }
    }
}

impl From<OptimizerKind> for OptimizerType {
    fn from(kind: OptimizerKind) -> Self {
        Self::for_kind(kind, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{Adam, Optimizer};

    #[test]
    fn adam_set_lr_updates_learning_rate() {
        let mut opt = Adam::new(0.01, 0.9, 0.999, 1e-8);
        opt.set_lr(0.001);
    }
}
