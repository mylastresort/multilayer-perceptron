use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};

#[test]
#[should_panic(expected = "at least 2 hidden layers")]
fn network_builder_requires_two_hidden_layers() {
    let _ = Network::new()
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
fn network_builder_allows_two_hidden_layers() {
    let network = Network::new()
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
