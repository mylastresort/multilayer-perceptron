use ndarray::{Array1, Array2};
use serde::Deserialize;

use crate::network::model::Network;
use crate::training::backprop::LayerGradients;

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
    },
}

pub trait Optimizer {
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]);
}

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    /// update: W ← W − η·∂L/∂W, b ← b − η·∂L/∂b.
    fn update(&mut self, network: &mut Network, grads: &[LayerGradients]) {
        assert_eq!(
            network.layers.len(),
            grads.len(),
            "network layer count and gradient count must match"
        );
        for (layer, g) in network.layers.iter_mut().zip(grads.iter()) {
            layer.weights = &layer.weights - &(g.weights.clone() * self.lr);
            layer.bias = &layer.bias - &(g.bias.clone() * self.lr);
        }
    }
}

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m: Vec<(Array2<f64>, Array1<f64>)>,
    v: Vec<(Array2<f64>, Array1<f64>)>,
}

impl Adam {
    pub fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    fn ensure_moments(&mut self, network: &Network) {
        if self.m.is_empty() {
            self.m = network
                .layers
                .iter()
                .map(|l| {
                    (
                        Array2::zeros(l.weights.raw_dim()),
                        Array1::zeros(l.bias.raw_dim()),
                    )
                })
                .collect();
            self.v = self.m.clone();
        }
    }

    /// first moment update: m ← β₁·m + (1 − β₁)·g.
    fn update_first_moment(&mut self, i: usize, g: &LayerGradients) {
        let (mw, mb) = &mut self.m[i];
        *mw = &*mw * self.beta1 + &(g.weights.clone() * (1.0 - self.beta1));
        *mb = &*mb * self.beta1 + &(g.bias.clone() * (1.0 - self.beta1));
    }

    /// second moment update: v ← β₂·v + (1 − β₂)·g².
    fn update_second_moment(&mut self, i: usize, g: &LayerGradients) {
        let (vw, vb) = &mut self.v[i];
        *vw = &*vw * self.beta2 + &(g.weights.mapv(|x| x * x) * (1.0 - self.beta2));
        *vb = &*vb * self.beta2 + &(g.bias.mapv(|x| x * x) * (1.0 - self.beta2));
    }

    /// bias-corrected moment update: θ ← θ − η·(m/(1−β₁ᵗ))/(√(v/(1−β₂ᵗ)) + ε) for θ = W, b.
    fn apply_bias_corrected_update(
        &self,
        i: usize,
        w: &mut Array2<f64>,
        b: &mut Array1<f64>,
        bc1: f64,
        bc2: f64,
    ) {
        let (mw, mb) = &self.m[i];
        let (vw, vb) = &self.v[i];
        // bias-corrected moments: m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ)
        let (mw_hat, mb_hat) = (mw / bc1, mb / bc1);
        let (vw_hat, vb_hat) = (vw / bc2, vb / bc2);
        // parameter update: θ ← θ − η·m̂/(√v̂ + ε) for θ = W, b
        *w = &*w - &(mw_hat * self.lr / vw_hat.mapv(|v| v.sqrt() + self.epsilon));
        *b = &*b - &(mb_hat * self.lr / vb_hat.mapv(|v| v.sqrt() + self.epsilon));
    }

    /// bias correction factor for the first moment: bc1 = 1 − β₁ᵗ.
    fn bias_correction1(&self) -> f64 {
        1.0 - self.beta1.powi(self.t as i32)
    }

    /// bias correction factor for the second moment: bc2 = 1 − β₂ᵗ.
    fn bias_correction2(&self) -> f64 {
        1.0 - self.beta2.powi(self.t as i32)
    }
}

impl Optimizer for Adam {
    /// update: W ← W − η·m̂/(√v̂+ε), b ← b − η·m̂/(√v̂+ε) with bias-corrected moments m̂, v̂.
    fn update(&mut self, network: &mut Network, grads: &[LayerGradients]) {
        assert_eq!(network.layers.len(), grads.len());
        self.ensure_moments(network);

        self.t += 1;
        let bc1 = self.bias_correction1();
        let bc2 = self.bias_correction2();

        for (i, (layer, g)) in network.layers.iter_mut().zip(grads.iter()).enumerate() {
            self.update_first_moment(i, g);
            self.update_second_moment(i, g);
            self.apply_bias_corrected_update(i, &mut layer.weights, &mut layer.bias, bc1, bc2);
        }
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
            } => Box::new(Adam::new(learning_rate, *beta1, *beta2, *epsilon)),
        }
    }

    pub fn for_kind(kind: OptimizerKind) -> Self {
        match kind {
            OptimizerKind::Sgd => OptimizerType::SGD,
            OptimizerKind::Adam => OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        }
    }
}

impl From<OptimizerKind> for OptimizerType {
    fn from(kind: OptimizerKind) -> Self {
        Self::for_kind(kind)
    }
}
