use ndarray::{Array1, Array2, ArrayView2};

use crate::network::{
    activation::{Activation, ActivationFunction},
    initializer::WeightInitializer,
};

// Defines the Layer struct, which represents a single layer in the neural network,
// including its weights, bias, activation function, and caches for backpropagation.
pub struct Layer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub activation: ActivationFunction,
    // Cache for backpropagation
    pub input_cache: Option<Array2<f64>>,
    pub weighted_sum_cache: Option<Array2<f64>>,
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
            weighted_sum_cache: None,
        }
    }

    // Performs the forward pass through the layer.
    pub fn forward(&mut self, _input: ArrayView2<'_, f64>) -> Array2<f64> {
        let weighted_sum = _input.dot(&self.weights) + &self.bias;
        self.input_cache = Some(_input.to_owned());
        self.weighted_sum_cache = Some(weighted_sum.clone());
        self.activation.forward(&weighted_sum)
    }

    // Performs the backward pass through the layer, calculating gradients for weights, bias, and input.
    pub fn backward(&self, _grad_output: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let weighted_sum = self
            .weighted_sum_cache
            .as_ref()
            .expect("No forward pass cache");
        let input = self.input_cache.as_ref().expect("No forward pass cache");

        let d_a = self.activation.backward(weighted_sum, _grad_output);
        let d_z = input.t().dot(&d_a);
        let dw = d_a.sum_axis(ndarray::Axis(0));
        let db = d_a.dot(&self.weights.t());

        (db, d_z, dw)
    }
}
