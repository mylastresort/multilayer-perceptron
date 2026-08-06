use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
use ndarray::Array2;

fn build_network_with_known_weights() -> Network {
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

// --- Round-trip: save then load recovers identical weights ---

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

// --- Loaded model produces the same predictions as the original ---

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

// --- Error cases ---

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

fn three_layer_net() -> Network {
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
            ActivationFunction::Tanh,
            WeightInitializer::Xavier,
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
fn load_nonexistent_file_returns_error_from_persist() {
    let result = Network::load("/tmp/this_file_does_not_exist_mlp.json");
    assert!(result.is_err());
}

#[test]
fn load_invalid_json_returns_error_from_persist() {
    let path =
        std::env::temp_dir().join(format!("mlp_invalid_json_{}.json", std::process::id()));
    std::fs::write(&path, "not valid json {{").unwrap();
    let result = Network::load(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
}

#[test]
fn load_model_with_fewer_than_three_layers_returns_error() {
    let json = serde_json::json!({
        "learning_rate": 0.01,
        "layers": [
            {
                "weights": [[0.1, 0.2]],
                "bias": [0.0, 0.0],
                "activation": "sigmoid"
            },
            {
                "weights": [[0.3], [0.4]],
                "bias": [0.0],
                "activation": "sigmoid"
            }
        ]
    })
    .to_string();

    let path = std::env::temp_dir().join(format!("mlp_two_layers_{}.json", std::process::id()));
    std::fs::write(&path, json).unwrap();
    let result = Network::load(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
}

#[test]
fn load_layer_with_empty_weights_returns_error() {
    let json = serde_json::json!({
        "learning_rate": 0.01,
        "layers": [
            { "weights": [], "bias": [], "activation": "sigmoid" },
            { "weights": [[0.1]], "bias": [0.0], "activation": "sigmoid" },
            { "weights": [[0.2]], "bias": [0.0], "activation": "sigmoid" }
        ]
    })
    .to_string();

    let path =
        std::env::temp_dir().join(format!("mlp_empty_weights_{}.json", std::process::id()));
    std::fs::write(&path, json).unwrap();
    let result = Network::load(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
}

#[test]
fn load_layer_with_mismatched_bias_length_returns_error() {
    let json = serde_json::json!({
        "learning_rate": 0.01,
        "layers": [
            {
                "weights": [[0.1, 0.2], [0.3, 0.4]],
                "bias": [0.0, 0.0, 0.0],
                "activation": "sigmoid"
            },
            {
                "weights": [[0.3], [0.4]],
                "bias": [0.0],
                "activation": "sigmoid"
            },
            {
                "weights": [[0.5]],
                "bias": [0.0],
                "activation": "sigmoid"
            }
        ]
    })
    .to_string();

    let path =
        std::env::temp_dir().join(format!("mlp_bias_mismatch_{}.json", std::process::id()));
    std::fs::write(&path, json).unwrap();
    let result = Network::load(&path);
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let Err(e) = result else { unreachable!() };
    let msg = e.to_string();
    assert!(msg.contains("bias"), "unexpected error: {msg}");
}

#[test]
fn save_and_load_roundtrip_three_layer_net() {
    let net = three_layer_net();
    let path =
        std::env::temp_dir().join(format!("mlp_persist_unit_{}.json", std::process::id()));
    net.save(&path).unwrap();
    let loaded = Network::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.layers.len(), 3);
    assert!((loaded.learning_rate - net.learning_rate).abs() < 1e-12);
}

#[test]
fn save_and_load_preserves_loss_function() {
    use mlp::training::loss::LossFunction;
    let mut net = three_layer_net();
    net.loss = LossFunction::BinaryCrossEntropy;
    let path =
        std::env::temp_dir().join(format!("mlp_persist_loss_{}.json", std::process::id()));
    net.save(&path).unwrap();
    let loaded = Network::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.loss, LossFunction::BinaryCrossEntropy);
}

#[test]
fn load_model_without_loss_defaults_to_binary_cross_entropy() {
    use mlp::training::loss::LossFunction;
    let json = serde_json::json!({
        "learning_rate": 0.01,
        "layers": [
            { "weights": [[0.1, 0.2]], "bias": [0.0, 0.0], "activation": "sigmoid" },
            { "weights": [[0.3, 0.4], [0.5, 0.6]], "bias": [0.0, 0.0], "activation": "sigmoid" },
            { "weights": [[0.7], [0.8]], "bias": [0.0], "activation": "softmax" }
        ]
    })
    .to_string();

    let path = std::env::temp_dir().join(format!("mlp_no_loss_{}.json", std::process::id()));
    std::fs::write(&path, json).unwrap();
    let loaded = Network::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.loss, LossFunction::BinaryCrossEntropy);
}
