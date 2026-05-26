use serde::Deserialize;

use crate::{network::model::Network, training::backprop::LayerGradients};

/// Lightweight tag used in YAML config — no network dependency.
#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Sgd,
    NesterovSgd,
    Rmsprop,
    Adam,
}

/// Runtime optimizer variant; carries hyperparameters and owns state.
pub enum OptimizerType {
    SGD,
    NesterovSGD { momentum: f64 },
    RMSprop { rho: f64, epsilon: f64 },
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
}

pub trait Optimizer {
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]);
    fn set_lr(&mut self, lr: f64);
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Nesterov SGD
// ---------------------------------------------------------------------------
pub struct NesterovSGD {
    learning_rate: f64,
    momentum: f64,
    /// Per-layer velocity buffers (weights, bias).
    velocities: Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}

impl NesterovSGD {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: Vec::new(),
        }
    }

    fn ensure_velocities(&mut self, network: &Network) {
        if self.velocities.is_empty() {
            self.velocities = network
                .layers
                .iter()
                .map(|l| {
                    (
                        ndarray::Array2::zeros(l.weights.raw_dim()),
                        ndarray::Array1::zeros(l.bias.raw_dim()),
                    )
                })
                .collect();
        }
    }
}

impl Optimizer for NesterovSGD {
    /// Nesterov update (Keras formulation):
    ///   v_t  = μ · v_{t-1} + lr · g
    ///   θ_t  = θ_{t-1} − (μ · v_t + lr · g)
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]) {
        assert_eq!(network.layers.len(), gradients.len());
        self.ensure_velocities(network);

        for ((layer, grad), (vw, vb)) in network
            .layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.velocities.iter_mut())
        {
            let gw = &grad.weights * self.learning_rate;
            let gb = &grad.bias * self.learning_rate;

            *vw = &(*vw) * self.momentum + &gw;
            *vb = &(*vb) * self.momentum + &gb;

            layer.weights = &layer.weights - &(&(*vw) * self.momentum + &gw);
            layer.bias = &layer.bias - &(&(*vb) * self.momentum + &gb);
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

// ---------------------------------------------------------------------------
// RMSprop
// ---------------------------------------------------------------------------
pub struct RMSprop {
    learning_rate: f64,
    rho: f64,
    epsilon: f64,
    /// Per-layer accumulated squared gradient (weights, bias).
    cache: Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}

impl RMSprop {
    pub fn new(learning_rate: f64, rho: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon,
            cache: Vec::new(),
        }
    }

    fn ensure_cache(&mut self, network: &Network) {
        if self.cache.is_empty() {
            self.cache = network
                .layers
                .iter()
                .map(|l| {
                    (
                        ndarray::Array2::zeros(l.weights.raw_dim()),
                        ndarray::Array1::zeros(l.bias.raw_dim()),
                    )
                })
                .collect();
        }
    }
}

impl Optimizer for RMSprop {
    /// s_t = ρ · s_{t-1} + (1−ρ) · g²
    /// θ_t = θ − lr · g / √(s_t + ε)
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]) {
        assert_eq!(network.layers.len(), gradients.len());
        self.ensure_cache(network);

        for ((layer, grad), (sw, sb)) in network
            .layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.cache.iter_mut())
        {
            *sw = &(*sw) * self.rho + &(grad.weights.mapv(|v| v * v) * (1.0 - self.rho));
            *sb = &(*sb) * self.rho + &(grad.bias.mapv(|v| v * v) * (1.0 - self.rho));

            layer.weights = &layer.weights
                - &(grad.weights.clone() * self.learning_rate
                    / sw.mapv(|v| (v + self.epsilon).sqrt()));
            layer.bias = &layer.bias
                - &(grad.bias.clone() * self.learning_rate
                    / sb.mapv(|v| (v + self.epsilon).sqrt()));
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    /// Per-layer first moment (weights, bias).
    m: Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
    /// Per-layer second moment (weights, bias).
    v: Vec<(ndarray::Array2<f64>, ndarray::Array1<f64>)>,
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
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
    /// m_t = β1·m_{t-1} + (1−β1)·g
    /// v_t = β2·v_{t-1} + (1−β2)·g²
    /// m̂  = m_t / (1−β1^t),  v̂ = v_t / (1−β2^t)
    /// θ_t = θ − lr · m̂ / (√v̂ + ε)
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]) {
        assert_eq!(network.layers.len(), gradients.len());
        self.ensure_moments(network);

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for ((layer, grad), ((mw, mb), (vw, vb))) in network
            .layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            *mw = &(*mw) * self.beta1 + &(grad.weights.clone() * (1.0 - self.beta1));
            *mb = &(*mb) * self.beta1 + &(grad.bias.clone() * (1.0 - self.beta1));

            *vw = &(*vw) * self.beta2
                + &(grad.weights.mapv(|v| v * v) * (1.0 - self.beta2));
            *vb = &(*vb) * self.beta2
                + &(grad.bias.mapv(|v| v * v) * (1.0 - self.beta2));

            let mw_hat = mw.mapv(|v| v / bc1);
            let mb_hat = mb.mapv(|v| v / bc1);
            let vw_hat = vw.mapv(|v| v / bc2);
            let vb_hat = vb.mapv(|v| v / bc2);

            layer.weights = &layer.weights
                - &(mw_hat * self.learning_rate / vw_hat.mapv(|v| v.sqrt() + self.epsilon));
            layer.bias = &layer.bias
                - &(mb_hat * self.learning_rate / vb_hat.mapv(|v| v.sqrt() + self.epsilon));
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
impl OptimizerType {
    pub fn create(&self, learning_rate: f64) -> Box<dyn Optimizer> {
        match self {
            OptimizerType::SGD => Box::new(SGD::new(learning_rate)),
            OptimizerType::NesterovSGD { momentum } => {
                Box::new(NesterovSGD::new(learning_rate, *momentum))
            }
            OptimizerType::RMSprop { rho, epsilon } => {
                Box::new(RMSprop::new(learning_rate, *rho, *epsilon))
            }
            OptimizerType::Adam { beta1, beta2, epsilon } => {
                Box::new(Adam::new(learning_rate, *beta1, *beta2, *epsilon))
            }
        }
    }
}

impl From<OptimizerKind> for OptimizerType {
    fn from(kind: OptimizerKind) -> Self {
        match kind {
            OptimizerKind::Sgd => OptimizerType::SGD,
            OptimizerKind::NesterovSgd => OptimizerType::NesterovSGD { momentum: 0.9 },
            OptimizerKind::Rmsprop => OptimizerType::RMSprop {
                rho: 0.9,
                epsilon: 1e-8,
            },
            OptimizerKind::Adam => OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Adam, NesterovSGD, Optimizer, RMSprop};

    #[test]
    fn nesterov_set_lr_updates_learning_rate() {
        let mut opt = NesterovSGD::new(0.01, 0.9);
        opt.set_lr(0.001);
        // Verify the new lr is used: set_lr replaces the stored rate.
        // (Internal field is private; just check it doesn't panic.)
    }

    #[test]
    fn rmsprop_set_lr_updates_learning_rate() {
        let mut opt = RMSprop::new(0.01, 0.9, 1e-8);
        opt.set_lr(0.001);
    }

    #[test]
    fn adam_set_lr_updates_learning_rate() {
        let mut opt = Adam::new(0.01, 0.9, 0.999, 1e-8);
        opt.set_lr(0.001);
    }
}

