use mlp::app::predict::{PredictArgs, run_predict};
use mlp::data::preprocessing::{Normalizer, StandardScaler};
use mlp::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
use ndarray::s;

fn fitted_scaler() -> StandardScaler {
    let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));
    let dataset = mlp::app::training::build_dataset(&dataset_path).unwrap();
    let mut scaler = StandardScaler::default();
    scaler.fit(&dataset.features.slice(s![.., 1..]).to_owned());
    scaler
}

#[test]
fn run_predict_returns_error_for_missing_model() {
    let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));
    let result = run_predict(&PredictArgs {
        dataset_path,
        model_path: "/tmp/mlp_nonexistent_model_xyz123.json".to_string(),
    });
    assert!(result.is_err());
}

#[test]
fn run_predict_returns_error_for_empty_dataset() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let csv_path = format!("/tmp/mlp_empty_{}_{}.csv", std::process::id(), ts);
    std::fs::write(&csv_path, "id,diagnosis,f1\n").unwrap();

    let model_path = format!("/tmp/mlp_pred_empty_{}_{}.json", std::process::id(), ts);
    let network = Network::builder()
        .learning_rate(0.01)
        .add_layer(Layer::new(
            1,
            2,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            2,
            2,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            2,
            1,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .build();
    network.save(&model_path).expect("model should save");

    let result = run_predict(&PredictArgs {
        dataset_path: csv_path.clone(),
        model_path: model_path.clone(),
    });
    let _ = std::fs::remove_file(&csv_path);
    let _ = std::fs::remove_file(&model_path);
    assert!(result.is_err());
}

#[test]
fn run_predict_succeeds_with_single_output_network() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let model_path = format!("/tmp/mlp_predict_single_{}_{}.json", std::process::id(), ts);
    let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));

    let mut network = Network::builder()
        .learning_rate(0.01)
        .add_layer(Layer::new(
            30,
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
        .build();
    network.scaler = Some(fitted_scaler());
    network.save(&model_path).expect("model should save");

    let result = run_predict(&PredictArgs {
        dataset_path,
        model_path: model_path.clone(),
    });
    let _ = std::fs::remove_file(&model_path);
    assert!(result.is_ok(), "run_predict failed: {:?}", result.err());
}

#[test]
fn run_predict_succeeds_with_two_output_network() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let model_path = format!("/tmp/mlp_predict_two_{}_{}.json", std::process::id(), ts);
    let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));

    let mut network = Network::builder()
        .learning_rate(0.01)
        .add_layer(Layer::new(
            30,
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
    network.scaler = Some(fitted_scaler());
    network.save(&model_path).expect("model should save");

    let result = run_predict(&PredictArgs {
        dataset_path,
        model_path: model_path.clone(),
    });
    let _ = std::fs::remove_file(&model_path);
    assert!(result.is_ok(), "run_predict failed: {:?}", result.err());
}
