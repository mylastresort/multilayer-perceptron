use crate::{network::model::Network, training::backprop::LayerGradients};

pub enum OptimizerType {
    SGD,
    // Future optimizers can be added here (e.g., Adam, RMSProp)
}

pub trait Optimizer {
    fn update(&mut self, network: &mut Network, gradients: &[LayerGradients]);
    fn set_lr(&mut self, lr: f64);
}

pub struct SGD {
    learning_rate: f64,
}

// implement a factory pattern for creating optimizers based on the OptimizerType enum
impl OptimizerType {
    pub fn create(&self, learning_rate: f64) -> Box<dyn Optimizer> {
        match self {
            OptimizerType::SGD => Box::new(SGD::new(learning_rate)),
            // Future optimizers can be added here
        }
    }
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
