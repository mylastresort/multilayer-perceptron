use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
use mlp::training::backprop::LayerGradients;
use mlp::training::optimizer::{Adam, Optimizer, SGD};
use ndarray::{Array1, Array2};

fn tiny_network() -> Network {
    Network::new()
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

fn zero_gradients(network: &Network) -> Vec<LayerGradients> {
    network
        .layers
        .iter()
        .map(|l| LayerGradients {
            weights: Array2::zeros(l.weights.raw_dim()),
            bias: Array1::zeros(l.bias.raw_dim()),
        })
        .collect()
}

fn unit_gradients(network: &Network) -> Vec<LayerGradients> {
    network
        .layers
        .iter()
        .map(|l| LayerGradients {
            weights: Array2::ones(l.weights.raw_dim()),
            bias: Array1::ones(l.bias.raw_dim()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------

#[test]
fn sgd_zero_gradient_leaves_weights_unchanged() {
    let mut net = tiny_network();
    let w0_before = net.layers[0].weights.clone();

    let grads = zero_gradients(&net);
    let mut opt = SGD::new(0.1);
    opt.update(&mut net, &grads);

    assert_eq!(net.layers[0].weights, w0_before);
}

#[test]
fn sgd_unit_gradient_decreases_weights_by_lr() {
    let mut net = tiny_network();
    let w0_before = net.layers[0].weights.clone();

    let grads = unit_gradients(&net);
    let mut opt = SGD::new(0.1);
    opt.update(&mut net, &grads);

    let expected = w0_before - 0.1_f64;
    for ((i, j), v) in net.layers[0].weights.indexed_iter() {
        let diff = (v - expected[[i, j]]).abs();
        assert!(
            diff < 1e-12,
            "weight [{i},{j}] mismatch: {v} vs {}",
            expected[[i, j]]
        );
    }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

#[test]
fn adam_zero_gradient_leaves_weights_unchanged() {
    let mut net = tiny_network();
    let w0_before = net.layers[0].weights.clone();

    let grads = zero_gradients(&net);
    let mut opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
    opt.update(&mut net, &grads);

    // m and v stay zero → update = 0.
    assert_eq!(net.layers[0].weights, w0_before);
}

#[test]
fn adam_unit_gradient_updates_weights() {
    let mut net = tiny_network();
    let w0_before = net.layers[0].weights.clone();

    let grads = unit_gradients(&net);
    let mut opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
    opt.update(&mut net, &grads);

    assert_ne!(net.layers[0].weights, w0_before);
}

#[test]
fn adam_bias_correction_at_step1() {
    // At t=1, bias-corrected first moment m̂ = g/(1-β1) and second moment v̂ = g²/(1-β2).
    // With unit gradients and lr=1: Δw = 1/(sqrt(1/(1-β2)) + ε).
    let mut net = tiny_network();

    // Zero-out all weights for a clean reference.
    for layer in net.layers.iter_mut() {
        layer.weights.fill(0.0);
        layer.bias.fill(0.0);
    }

    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let epsilon = 1e-8_f64;
    let lr = 1.0_f64;

    let grads = unit_gradients(&net);
    let mut opt = Adam::new(lr, beta1, beta2, epsilon);
    opt.update(&mut net, &grads);

    // At t=1 with unit gradients g=1:
    //   m1 = (1−β1)·g = 0.1         m̂1 = m1/(1−β1^1) = 0.1/0.1 = 1.0
    //   v1 = (1−β2)·g² = 0.001      v̂1 = v1/(1−β2^1) = 0.001/0.001 = 1.0
    //   Δw = lr · m̂1 / (√v̂1 + ε) = 1.0 / (1.0 + ε) ≈ 1.0
    let m_hat = 1.0_f64; // simplifies to 1.0 for unit gradient
    let v_hat = 1.0_f64;
    let expected_delta = lr * m_hat / (v_hat.sqrt() + epsilon);

    for (i, j) in [(0usize, 0usize)] {
        let actual = net.layers[0].weights[[i, j]];
        let expected = -expected_delta;
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-9,
            "adam step-1 weight [{i},{j}]: got {actual}, expected {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// OptimizerType factory
// ---------------------------------------------------------------------------

#[test]
fn optimizer_type_factory_creates_correct_variant() {
    use mlp::training::optimizer::{OptimizerKind, OptimizerType};

    let lr = 0.01;

    // Each factory call must return a working optimizer (smoke test).
    let kinds = [OptimizerKind::Sgd, OptimizerKind::Adam];
    for kind in kinds {
        let opt_type = OptimizerType::from(kind);
        let mut opt = opt_type.create(lr);
        let mut net = tiny_network();
        let grads = unit_gradients(&net);
        // Must not panic.
        opt.update(&mut net, &grads);
    }
}
