use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
use mlp::training::backprop::LayerGradients;
use mlp::training::loss::LossFunction;
use ndarray::{Array1, Array2, arr1, arr2};

fn tiny_net() -> Network {
    Network::builder()
        .add_layer(Layer::new(
            2,
            4,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            4,
            4,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            4,
            1,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .build()
}

#[test]
fn layer_gradients_from_tuple() {
    let w = Array2::zeros((3, 4));
    let b = Array1::zeros(4);
    let lg = LayerGradients::from((w.clone(), b.clone()));
    assert_eq!(lg.weights.dim(), w.dim());
    assert_eq!(lg.bias.len(), b.len());
}

#[test]
fn backward_returns_one_gradient_per_layer() {
    let mut net = tiny_net();
    let input = arr2(&[[0.5, 0.3]]);
    let _ = net.forward(input.view());
    let grads = net.backward(LossFunction::BinaryCrossEntropy, arr1(&[0.0f64]).view());
    assert_eq!(grads.len(), net.layers.len());
}

#[test]
fn backward_gradient_shapes_match_layer_weights() {
    let mut net = tiny_net();
    let input = arr2(&[[0.5, 0.3]]);
    let _ = net.forward(input.view());
    let grads = net.backward(LossFunction::BinaryCrossEntropy, arr1(&[0.0f64]).view());
    for (layer, grad) in net.layers.iter().zip(grads.iter()) {
        assert_eq!(grad.weights.dim(), layer.weights.dim());
        assert_eq!(grad.bias.len(), layer.bias.len());
    }
}

fn bce_loss_of(net: &mut Network, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
    let p = net.forward(x.view());
    LossFunction::BinaryCrossEntropy
        .compute(p.view(), y.view())
        .mean()
        .unwrap()
}

fn assert_weights_match_finite_difference(
    net: &mut Network,
    layer_idx: usize,
    analytic: &Array2<f64>,
    eps: f64,
    loss_of: &dyn Fn(&mut Network) -> f64,
) {
    let weights_dim = net.layers[layer_idx].weights.dim();
    for i in 0..weights_dim.0 {
        for j in 0..weights_dim.1 {
            let old = net.layers[layer_idx].weights[[i, j]];
            net.layers[layer_idx].weights[[i, j]] = old + eps;
            let l_plus = loss_of(net);
            net.layers[layer_idx].weights[[i, j]] = old - eps;
            let l_minus = loss_of(net);
            net.layers[layer_idx].weights[[i, j]] = old;

            let numeric = (l_plus - l_minus) / (2.0 * eps);
            let value = analytic[[i, j]];
            let diff = (value - numeric).abs();
            assert!(
                diff < 1e-5,
                "layer {layer_idx} w[{i},{j}]: analytic={value:.8} numeric={numeric:.8} diff={diff:.8}"
            );
        }
    }
}

fn assert_backward_matches_finite_difference(output_activation: ActivationFunction, n_out: usize) {
    let mut net = Network::builder()
        .add_layer(Layer::new(
            2,
            4,
            ActivationFunction::Sigmoid,
            WeightInitializer::Xavier,
        ))
        .add_layer(Layer::new(
            4,
            4,
            ActivationFunction::Tanh,
            WeightInitializer::Xavier,
        ))
        .add_layer(Layer::new(4, n_out, output_activation, WeightInitializer::Xavier))
        .build();
    for l in net.layers.iter_mut() {
        l.weights = Array2::from_shape_fn(l.weights.dim(), |(i, j)| {
            0.02 * (i as f64 + 2.0 * j as f64 + 1.0)
        });
        l.bias = Array1::from_elem(l.bias.len(), 0.1);
    }

    let x = arr2(&[[0.5, 1.0], [1.5, 0.3], [0.8, 1.2]]);
    let y = arr1(&[1.0, 0.0, 1.0]);

    let _ = net.forward(x.view());
    let grads = net.backward(LossFunction::BinaryCrossEntropy, y.view());

    let loss_of = |net: &mut Network| bce_loss_of(net, &x, &y);
    for (layer_idx, grad) in grads.iter().enumerate() {
        assert_weights_match_finite_difference(&mut net, layer_idx, &grad.weights, 1e-6, &loss_of);
    }
}

#[test]
fn bce_backward_matches_finite_difference_sigmoid_output() {
    assert_backward_matches_finite_difference(ActivationFunction::Sigmoid, 1);
    assert_backward_matches_finite_difference(ActivationFunction::Sigmoid, 2);
}

#[test]
fn bce_backward_matches_finite_difference_tanh_output() {
    assert_backward_matches_finite_difference(ActivationFunction::Tanh, 2);
}

#[test]
fn bce_backward_matches_finite_difference_relu_output() {
    assert_backward_matches_finite_difference(ActivationFunction::ReLU, 2);
}

#[test]
fn bce_backward_matches_finite_difference_softmax_output() {
    assert_backward_matches_finite_difference(ActivationFunction::Softmax, 2);
}
