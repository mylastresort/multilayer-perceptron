use ndarray::{Array1, Array2, ArrayView2};

use crate::network::{
    activation::{Activation, ActivationFunction},
    initializer::WeightInitializer,
};

pub struct Layer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub activation: ActivationFunction,
    pub input_cache: Option<Array2<f64>>,
    pub activated_cache: Option<Array2<f64>>,
}

impl Layer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        initializer: WeightInitializer,
    ) -> Self {
        let weights = initializer.initialize(input_size, output_size);
        let bias = Array1::zeros(output_size);
        Self {
            weights,
            bias,
            activation,
            input_cache: None,
            activated_cache: None,
        }
    }

    /// forward: z = input·W + b, a = φ(z). Caches input (a_prev) and a for backward.
    pub fn forward(&mut self, input: ArrayView2<'_, f64>) -> Array2<f64> {
        let z = input.dot(&self.weights) + &self.bias;
        self.input_cache = Some(input.to_owned());
        let a = self.activation.forward(&z);
        self.activated_cache = Some(a.clone());
        a
    }

    /// backward: Δ = ∂L/∂z = φ.backward(a, ∂L/∂a) for element-wise φ (hidden layers); returns (∂L/∂a_prev, ∂L/∂W, ∂L/∂b).
    pub fn backward(&self, upstream: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let (input, a) = self.caches();
        let delta = self.activation.backward(a, upstream);
        self.gradients(input, &delta)
    }

    /// backward_with_delta: output-layer delta p − y (no gating); returns (∂L/∂a_prev, ∂L/∂W, ∂L/∂b).
    pub fn backward_with_delta(
        &self,
        delta: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let input = self.input_cache.as_ref().expect("No forward pass cache");
        self.gradients(input, delta)
    }

    fn caches(&self) -> (&Array2<f64>, &Array2<f64>) {
        let input = self.input_cache.as_ref().expect("No forward pass cache");
        let a = self
            .activated_cache
            .as_ref()
            .expect("No forward pass cache");
        (input, a)
    }

    fn gradients(
        &self,
        input: &Array2<f64>,
        delta: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let grad_input = delta.dot(&self.weights.t());
        let grad_weights = input.t().dot(delta);
        let grad_bias = delta.sum_axis(ndarray::Axis(0));

        (grad_input, grad_weights, grad_bias)
    }
}
