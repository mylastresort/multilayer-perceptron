use mlp::app::training::{build_dataset, prepare_training_data, train_from_dataset};
use mlp::app::types::MonitorOptions;
use mlp::data::loader::Dataset;
use mlp::network::config::NetworkConfig;
use ndarray::{Array1, Array2};
use std::path::Path;

fn data_csv_path() -> String {
    format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"))
}

fn one_epoch_config() -> NetworkConfig {
    let yaml = r#"
learning_rate: 0.01
epochs: 1
batch_size: 32
input_layers:
  - size: 30
hidden_layers:
  - size: 8
  - size: 8
output_layers:
  - size: 2
"#;
    serde_yaml::from_str(yaml).expect("minimal config should parse")
}

#[test]
fn build_dataset_loads_real_csv() {
    let result = build_dataset(&data_csv_path());
    assert!(result.is_ok(), "build_dataset failed: {:?}", result.err());
    let dataset = result.unwrap();
    assert!(dataset.features.nrows() > 100);
}

#[test]
fn build_dataset_returns_error_for_missing_file() {
    let result = build_dataset("/tmp/mlp_nonexistent_data_xyz.csv");
    assert!(result.is_err());
}

#[test]
fn train_from_dataset_runs_one_epoch() {
    let path = data_csv_path();
    let dataset = build_dataset(&path).expect("data should load");
    let config = one_epoch_config();
    let result = train_from_dataset(&dataset, &config, &MonitorOptions::default(), None, None);
    assert!(result.is_ok(), "training failed: {:?}", result.err());
}

#[test]
fn train_from_dataset_rejects_dataset_with_fewer_than_three_rows() {
    let features = Array2::from_shape_fn((2, 31), |(i, j)| {
        if j == 0 {
            i as f64
        } else {
            (i + j) as f64 * 0.1
        }
    });
    let dataset = Dataset {
        features,
        labels: Array1::zeros(2),
        feature_names: Vec::new(),
    };
    let config = one_epoch_config();
    let result = train_from_dataset(&dataset, &config, &MonitorOptions::default(), None, None);
    let Err(e) = result else {
        panic!("expected Err for < 3 rows")
    };
    assert!(e.to_string().contains("at least 3 rows"), "unexpected: {e}");
}

#[test]
fn train_from_dataset_rejects_dataset_with_bad_split_ratios() {
    let features = Array2::from_shape_fn((3, 31), |(i, j)| {
        if j == 0 {
            (i % 2) as f64
        } else {
            (i + j) as f64 * 0.1
        }
    });
    let dataset = Dataset {
        features,
        labels: Array1::zeros(3),
        feature_names: Vec::new(),
    };
    let config = one_epoch_config();
    let result = train_from_dataset(&dataset, &config, &MonitorOptions::default(), None, None);
    let Err(e) = result else {
        panic!("expected Err for bad split")
    };
    assert!(
        e.to_string().contains("empty training or validation"),
        "unexpected: {e}"
    );
}

#[test]
fn train_from_dataset_saves_history_when_history_out_is_set() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let history_path = format!("/tmp/mlp_history_{}_{}.json", std::process::id(), ts);

    let dataset = build_dataset(&data_csv_path()).expect("data should load");
    let config = one_epoch_config();
    let opts = MonitorOptions {
        history_out: Some(history_path.clone()),
        ..MonitorOptions::default()
    };
    let result = train_from_dataset(&dataset, &config, &opts, None, None);
    let _ = std::fs::remove_file(&history_path);
    assert!(result.is_ok(), "training failed: {:?}", result.err());
}

#[test]
fn train_from_dataset_saves_model_when_model_out_is_set() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let model_path = format!("/tmp/mlp_model_{}_{}.json", std::process::id(), ts);

    let dataset = build_dataset(&data_csv_path()).expect("data should load");
    let config = one_epoch_config();
    let result = train_from_dataset(
        &dataset,
        &config,
        &MonitorOptions::default(),
        None,
        Some(Path::new(&model_path)),
    );
    let saved = result.is_ok() && Path::new(&model_path).exists();
    let _ = std::fs::remove_file(&model_path);
    assert!(saved, "model should have been saved");
}

#[test]
fn train_from_dataset_saves_curves_when_curves_out_is_set() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let curve_path = format!("/tmp/mlp_curves_{}_{}.png", std::process::id(), ts);

    let dataset = build_dataset(&data_csv_path()).expect("data should load");
    let config = one_epoch_config();
    let opts = MonitorOptions {
        curves_out: Some(curve_path.clone()),
        ..MonitorOptions::default()
    };
    let result = train_from_dataset(&dataset, &config, &opts, None, None);
    let saved = result.is_ok() && Path::new(&curve_path).exists();
    let _ = std::fs::remove_file(&curve_path);
    assert!(saved, "curves should have been saved");
}

#[test]
fn prepare_training_data_stratifies_on_diagnosis() {
    let dataset = build_dataset(&data_csv_path()).expect("data should load");
    let overall_positive = dataset
        .features
        .column(0)
        .iter()
        .filter(|v| **v >= 0.5)
        .count() as f64
        / dataset.features.nrows() as f64;

    let data = prepare_training_data(&dataset).expect("training data should prepare");

    assert!(
        data.y_train.iter().all(|v| *v == 0.0 || *v == 1.0),
        "train targets must be binary Diagnosis, got {:?}",
        data.y_train
    );
    assert!(
        (data.y_train.mean().unwrap() - overall_positive).abs() < 0.05,
        "train Diagnosis proportion {} drifted from overall {}",
        data.y_train.mean().unwrap(),
        overall_positive
    );
    assert!(
        (data.y_val.mean().unwrap() - overall_positive).abs() < 0.1,
        "val Diagnosis proportion {} drifted from overall {}",
        data.y_val.mean().unwrap(),
        overall_positive
    );
}
