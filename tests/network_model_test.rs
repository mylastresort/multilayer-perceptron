use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
    model::{FitConfig, Network},
};

#[test]
#[should_panic(expected = "at least 2 hidden layers")]
fn builder_rejects_less_than_two_hidden_layers() {
    let _ = Network::builder()
        .add_layer(Layer::new(
            30,
            24,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            24,
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();
}

#[test]
fn builder_accepts_two_hidden_layers_and_output() {
    let network = Network::builder()
        .add_layer(Layer::new(
            30,
            24,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            24,
            24,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            24,
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();

    assert_eq!(network.layers.len(), 3);
}

#[test]
fn network_forward_output_shape_is_correct() {
    let mut net = Network::builder()
        .add_layer(Layer::new(
            4,
            8,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            8,
            8,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            8,
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();

    let input = ndarray::Array2::zeros((5, 4));
    let output = net.forward(input.view());
    assert_eq!(output.dim(), (5, 2));
}

#[test]
fn network_predict_matches_forward() {
    let mut net = Network::builder()
        .add_layer(Layer::new(
            2,
            4,
            ActivationFunction::ReLU,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            4,
            4,
            ActivationFunction::ReLU,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            4,
            1,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .build();

    let input = ndarray::Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64 * 0.1);
    let out_fwd = net.forward(input.view());
    let out_pred = net.predict(input.view());
    assert_eq!(out_fwd, out_pred);
}

#[test]
fn network_learning_rate_builder_sets_field() {
    let net = Network::builder()
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
        .learning_rate(0.05)
        .build();
    assert!((net.learning_rate - 0.05).abs() < 1e-12);
}

#[test]
fn network_fit_one_epoch_returns_metrics() {
    use mlp::training::{loss::LossFunction, optimizer::OptimizerType};
    use ndarray::{Array1, Array2};

    let mut net = Network::builder()
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
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();

    let x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64 * 0.1);
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    let metrics = net.fit_with_callbacks(
        x.view(),
        y.view(),
        None,
        FitConfig {
            batch_size: 4,
            epochs: 1,
            optimizer: OptimizerType::SGD,
            loss_fn: LossFunction::BinaryCrossEntropy,
        },
        &mut [],
    );
    assert!(metrics.train_loss.is_finite());
}

#[test]
fn network_fit_accepts_array1_reference_as_target() {
    use mlp::training::{loss::LossFunction, optimizer::OptimizerType};
    use ndarray::{Array1, Array2};

    let mut net = Network::builder()
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
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();

    let x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64 * 0.1);
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    let metrics = net.fit_with_callbacks(
        x.view(),
        &y,
        None,
        FitConfig {
            batch_size: 4,
            epochs: 1,
            optimizer: OptimizerType::SGD,
            loss_fn: LossFunction::BinaryCrossEntropy,
        },
        &mut [],
    );
    assert!(metrics.train_loss.is_finite());
}
