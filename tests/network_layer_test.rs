use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
};
use ndarray::arr2;

fn simple_layer() -> Layer {
    let mut layer = Layer::new(2, 3, ActivationFunction::Sigmoid, WeightInitializer::He);
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
fn layer_forward_caches_input_and_activation() {
    let mut layer = simple_layer();
    let input = arr2(&[[1.0, 2.0]]);
    layer.forward(input.view());
    assert!(layer.input_cache.is_some());
    assert!(layer.activated_cache.is_some());
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
    let _ = layer.forward(input.view());
    let grad_output = arr2(&[[0.1, 0.2, 0.3]]);
    let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_output);
    assert_eq!(grad_input.dim(), (1, 2));
    assert_eq!(grad_weights.dim(), (2, 3));
    assert_eq!(grad_bias.len(), 3);
}

#[test]
fn layer_new_bias_is_zeros() {
    let layer = Layer::new(4, 8, ActivationFunction::ReLU, WeightInitializer::Xavier);
    assert!(layer.bias.iter().all(|&v| v == 0.0));
}
