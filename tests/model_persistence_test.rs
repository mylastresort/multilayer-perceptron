use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
use ndarray::Array2;

fn build_network_with_known_weights() -> Network {
    let mut net = Network::new()
        .add_layer(Layer::new(
            4,
            8,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            8,
            8,
            ActivationFunction::Tanh,
            WeightInitializer::Xavier,
        ))
        .add_layer(Layer::new(
            8,
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();

    // Overwrite weights with deterministic values so the round-trip is testable.
    for (idx, layer) in net.layers.iter_mut().enumerate() {
        layer.weights.indexed_iter_mut().for_each(|((i, j), v)| {
            *v = (idx as f64 + 1.0) * 0.1 + i as f64 * 0.01 + j as f64 * 0.001;
        });
        layer.bias.iter_mut().enumerate().for_each(|(i, v)| {
            *v = -(idx as f64 + 1.0) * 0.01 - i as f64 * 0.001;
        });
    }

    net
}

// ---------------------------------------------------------------------------
// Round-trip: save then load recovers identical weights
// ---------------------------------------------------------------------------

#[test]
fn save_load_roundtrip_preserves_weights() {
    let net = build_network_with_known_weights();
    let path = std::env::temp_dir().join("mlp_persist_test.json");

    net.save(&path).expect("save should succeed");
    let loaded = Network::load(&path).expect("load should succeed");

    assert_eq!(
        net.layers.len(),
        loaded.layers.len(),
        "layer count must match"
    );

    for (i, (orig, rest)) in net.layers.iter().zip(loaded.layers.iter()).enumerate() {
        assert_eq!(
            orig.weights.dim(),
            rest.weights.dim(),
            "weight shape of layer {i} must match"
        );
        for ((r, c), v) in orig.weights.indexed_iter() {
            let diff = (v - rest.weights[[r, c]]).abs();
            assert!(
                diff < 1e-14,
                "weight [{i}][{r},{c}] mismatch after round-trip: {v} vs {}",
                rest.weights[[r, c]]
            );
        }
        for (j, v) in orig.bias.iter().enumerate() {
            let diff = (v - rest.bias[j]).abs();
            assert!(diff < 1e-14, "bias [{i}][{j}] mismatch after round-trip");
        }
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn save_load_roundtrip_preserves_learning_rate() {
    let mut net = build_network_with_known_weights();
    net.learning_rate = 0.03141;

    let path = std::env::temp_dir().join("mlp_persist_lr_test.json");
    net.save(&path).expect("save should succeed");
    let loaded = Network::load(&path).expect("load should succeed");

    let diff = (loaded.learning_rate - 0.03141).abs();
    assert!(
        diff < 1e-12,
        "learning rate mismatch: {}",
        loaded.learning_rate
    );
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Loaded model produces the same predictions as the original
// ---------------------------------------------------------------------------

#[test]
fn loaded_model_produces_identical_predictions() {
    let mut net = build_network_with_known_weights();
    let path = std::env::temp_dir().join("mlp_persist_pred_test.json");

    let input = Array2::from_shape_vec(
        (3, 4),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    )
    .unwrap();

    let pred_before = net.predict(&input);
    net.save(&path).expect("save should succeed");

    let mut loaded = Network::load(&path).expect("load should succeed");
    let pred_after = loaded.predict(&input);

    assert_eq!(pred_before.dim(), pred_after.dim());
    for ((i, j), v) in pred_before.indexed_iter() {
        let diff = (v - pred_after[[i, j]]).abs();
        assert!(
            diff < 1e-12,
            "prediction mismatch at [{i},{j}]: {v} vs {}",
            pred_after[[i, j]]
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn load_nonexistent_file_returns_error() {
    let result = Network::load("/tmp/mlp_does_not_exist_xyz.json");
    assert!(result.is_err(), "loading a missing file should return Err");
}

#[test]
fn load_invalid_json_returns_error() {
    let path = std::env::temp_dir().join("mlp_persist_bad.json");
    std::fs::write(&path, b"{ not valid json ").unwrap();
    let result = Network::load(&path);
    assert!(result.is_err(), "loading invalid JSON should return Err");
    let _ = std::fs::remove_file(&path);
}
