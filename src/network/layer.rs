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
    pub fn backward(&self, upstream: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let weighted_sum = self
            .weighted_sum_cache
            .as_ref()
            .expect("No forward pass cache");
        let input = self.input_cache.as_ref().expect("No forward pass cache");

        let delta = self.activation.backward(weighted_sum, upstream);
        let g_input = delta.dot(&self.weights.t());
        let g_weights = input.t().dot(&delta);
        let g_bias = delta.sum_axis(ndarray::Axis(0));

        (g_input, g_weights, g_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::Layer;
    use crate::network::{activation::ActivationFunction, initializer::WeightInitializer};
    use ndarray::arr2;

    fn simple_layer() -> Layer {
        let mut layer = Layer::new(2, 3, ActivationFunction::Sigmoid, WeightInitializer::He);
        // Set deterministic weights so shapes are predictable
        layer.weights = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        layer.bias = ndarray::Array1::from(vec![0.0, 0.0, 0.0]);
        layer
    }

    #[test]
    fn layer_forward_output_shape_is_correct() {
        let mut layer = simple_layer();
        let input = arr2(&[[1.0, 2.0]]);
        let output = layer.forward(input.view());
        assert_eq!(output.dim(), (1, 3));
    }

    #[test]
    fn layer_forward_caches_input_and_weighted_sum() {
        let mut layer = simple_layer();
        let input = arr2(&[[1.0, 2.0]]);
        layer.forward(input.view());
        assert!(layer.input_cache.is_some());
        assert!(layer.weighted_sum_cache.is_some());
    }

    #[test]
    fn layer_forward_output_values_are_in_sigmoid_range() {
        let mut layer = simple_layer();
        let input = arr2(&[[1.0, 2.0]]);
        let output = layer.forward(input.view());
        for v in output.iter() {
            assert!(*v > 0.0 && *v < 1.0, "sigmoid output {v} outside (0, 1)");
        }
    }

    #[test]
    fn layer_backward_gradient_shapes_are_correct() {
        let mut layer = simple_layer();
        let input = arr2(&[[1.0, 2.0]]);
        let _ = layer.forward(input.view()); // populate cache
        let grad_output = arr2(&[[0.1, 0.2, 0.3]]);
        let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_output);
        assert_eq!(grad_input.dim(), (1, 2)); // same as input
        assert_eq!(grad_weights.dim(), (2, 3)); // same as weights
        assert_eq!(grad_bias.len(), 3); // same as bias
    }

    #[test]
    fn layer_new_bias_is_zeros() {
        let layer = Layer::new(4, 8, ActivationFunction::ReLU, WeightInitializer::Xavier);
        assert!(layer.bias.iter().all(|&v| v == 0.0));
    }
}
