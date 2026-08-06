use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
use mlp::training::loss::LossFunction;
use ndarray::{array, Array1, Array2};

fn build_net() -> Network {
    Network::builder()
        .learning_rate(0.1)
        .loss(LossFunction::BinaryCrossEntropy)
        .add_layer(Layer::new(2, 4, ActivationFunction::ReLU, WeightInitializer::Xavier))
        .add_layer(Layer::new(4, 3, ActivationFunction::Sigmoid, WeightInitializer::Xavier))
        .add_layer(Layer::new(3, 2, ActivationFunction::Softmax, WeightInitializer::Xavier))
        .build()
}

fn loss_at(net: &mut Network, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
    let pred = net.forward(x);
    LossFunction::BinaryCrossEntropy
        .compute(pred.view(), y.view())
        .mean()
        .unwrap()
}

#[test]
fn backprop_gradients_match_numerical_differentiation() {
    let mut net = build_net();
    let x = array![[0.5, -1.2], [1.3, 0.7], [-0.8, 0.9]];
    let y = array![0.0, 1.0, 0.0];

    let _ = net.forward(&x);
    let grads = net.backward(LossFunction::BinaryCrossEntropy, y.view());

    let h = 1e-6;
    let tolerance = 1e-4;

    for (li, layer_grad) in grads.iter().enumerate() {
        let (rows, cols) = layer_grad.weights.dim();
        for wi in 0..rows {
            for wj in 0..cols {
                let orig = net.layers[li].weights[[wi, wj]];
                net.layers[li].weights[[wi, wj]] = orig + h;
                let loss_plus = loss_at(&mut net, &x, &y);
                net.layers[li].weights[[wi, wj]] = orig - h;
                let loss_minus = loss_at(&mut net, &x, &y);
                net.layers[li].weights[[wi, wj]] = orig;
                let numerical = (loss_plus - loss_minus) / (2.0 * h);
                let analytical = layer_grad.weights[[wi, wj]];
                assert!(
                    (numerical - analytical).abs() < tolerance,
                    "layer {li} weight [{wi},{wj}]: analytical {analytical}, numerical {numerical}"
                );
            }
        }
        for bi in 0..layer_grad.bias.len() {
            let orig = net.layers[li].bias[bi];
            net.layers[li].bias[bi] = orig + h;
            let loss_plus = loss_at(&mut net, &x, &y);
            net.layers[li].bias[bi] = orig - h;
            let loss_minus = loss_at(&mut net, &x, &y);
            net.layers[li].bias[bi] = orig;
            let numerical = (loss_plus - loss_minus) / (2.0 * h);
            let analytical = layer_grad.bias[bi];
            assert!(
                (numerical - analytical).abs() < tolerance,
                "layer {li} bias [{bi}]: analytical {analytical}, numerical {numerical}"
            );
        }
    }
}
