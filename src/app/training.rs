use std::error::Error;
use std::path::Path;

use mlp::console::{Tone, bold, paint};
use mlp::data::loader::{Dataset, load_dataset};
use mlp::network::callbacks::{Callback, ProgressLogger};
use mlp::network::config::NetworkConfig;
use mlp::network::model::Network;
use mlp::training::monitor::{EarlyStoppingCallback, EarlyStoppingConfig, HistoryCallback};
use mlp::training::{loss::LossFunction, optimizer::OptimizerType};
use mlp::visualization::live_monitor::{GuiMonitorConfig, LiveTrainingMonitorCallback};
use ndarray::{Array2, Axis, s};

use super::types::MonitorOptions;

pub fn build_dataset(dataset_path: &str) -> Result<Dataset, Box<dyn Error>> {
    let base_features = vec![
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ];
    let stats = vec!["mean", "se", "extreme"];

    let mut names: Vec<String> = vec!["ID".to_string(), "Diagnosis".to_string()];
    for feature in &base_features {
        for stat in &stats {
            names.push(format!("{}_{}", feature, stat));
        }
    }

    // Keep loader defaults aligned with training_learning_curve_test.
    load_dataset(dataset_path, 1, names, 0)
}

fn standardize_from_train(
    x_train: &Array2<f64>,
    x_other: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let means = x_train
        .mean_axis(Axis(0))
        .expect("training features should not be empty");
    let stds = x_train.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));

    let x_train_scaled = (x_train - &means) / &stds;
    let x_other_scaled = (x_other - &means) / &stds;

    (x_train_scaled, x_other_scaled)
}

pub fn train_from_dataset(
    dataset: &Dataset,
    network_config: &NetworkConfig,
    gui: bool,
    monitor_options: &MonitorOptions,
    model_out: Option<&Path>,
) -> Result<Network, Box<dyn Error>> {
    // Baseline feature prep mirrors training_learning_curve_test.
    let x_raw = dataset.features.slice(s![.., 1..]).to_owned();
    let y = dataset
        .features
        .column(0)
        .mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });

    let n = x_raw.nrows();
    if n < 3 {
        return Err("dataset must contain at least 3 rows to split train/val/test".into());
    }

    let train_end = (0.85 * n as f64).round() as usize;
    let val_end = n;
    if train_end == 0 || val_end <= train_end {
        return Err("dataset split produced empty training or validation set".into());
    }

    let x_train_raw = x_raw.slice(s![0..train_end, ..]).to_owned();
    let x_val_raw = x_raw.slice(s![train_end..val_end, ..]).to_owned();
    let y_train = y.slice(s![0..train_end]).to_owned();
    let y_val = y.slice(s![train_end..val_end]).to_owned();
    let (x_train, x_val) = standardize_from_train(&x_train_raw, &x_val_raw);

    let means = x_train_raw
        .mean_axis(Axis(0))
        .expect("training features should not be empty");
    let stds = x_train_raw.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));

    let mut network = network_config.build_network();
    network.feature_mean = Some(means);
    network.feature_std = Some(stds);
    let epochs = network_config.epochs;
    let batch_size = network_config.batch_size;

    let mut monitor = LiveTrainingMonitorCallback::new(GuiMonitorConfig::from_env(
        gui,
        monitor_options.metrics.clone(),
    ));
    let mut history = HistoryCallback::new();
    let mut early_stopping = EarlyStoppingCallback::new(EarlyStoppingConfig {
        enabled: monitor_options.early_stopping,
        metric: monitor_options.monitor_metric,
        mode: monitor_options.monitor_mode,
        patience: monitor_options.monitor_patience,
        min_delta: monitor_options.monitor_min_delta,
        start_epoch: monitor_options.monitor_start_epoch,
        restore_best_weights: true,
    });
    let mut progress_logger = ProgressLogger::new(epochs);
    let mut callbacks: Vec<&mut dyn Callback> = vec![
        &mut monitor,
        &mut history,
        &mut early_stopping,
        &mut progress_logger,
    ];

    let metrics = network.fit_with_callbacks(
        x_train.view(),
        y_train.view(),
        Some((x_val.view(), y_val.view())),
        batch_size,
        epochs,
        OptimizerType::for_kind(network_config.optimizer, network_config.weight_decay),
        LossFunction::CategoricalCrossEntropy,
        &mut callbacks,
    );

    drop(callbacks);

    // Keras `restore_best_weights`: when early stopping is enabled the weights
    // are rolled back to the best monitored epoch, so the saved model is the
    // best one rather than the last epoch's. No-op when early stopping is off.
    early_stopping.restore_best(&mut network);

    println!(
        "{} {} - {} - {} - {}",
        bold(&paint("Training summary:", Tone::Success)),
        paint(
            &format!("train_loss={:.4}", metrics.train_loss),
            Tone::TrainMetric
        ),
        paint(
            &format!("val_loss={:.4}", metrics.val_loss),
            Tone::ValMetric
        ),
        paint(
            &format!("train_acc={:.4}", metrics.train_accuracy),
            Tone::TrainMetric
        ),
        paint(
            &format!("val_acc={:.4}", metrics.val_accuracy),
            Tone::ValMetric
        )
    );
    println!(
        "{} {}",
        paint("Monitor points:", Tone::Info),
        paint(&format!("{} epochs", monitor.history_len()), Tone::Accent)
    );
    println!(
        "{} {}",
        paint("History records:", Tone::Info),
        paint(
            &format!("{} epochs", history.history().epochs.len()),
            Tone::Accent
        )
    );
    if let Some(stopped_epoch) = early_stopping.stopped_epoch() {
        println!(
            "{} {} {}",
            paint("Early stopping triggered at epoch", Tone::Warn),
            paint(&(stopped_epoch + 1).to_string(), Tone::Warn),
            paint(
                &format!("(monitored: {})", monitor_options.monitor_metric.as_str()),
                Tone::Warn
            )
        );
    }

    if let Some(path) = &monitor_options.history_out {
        history.save_json(path)?;
        println!("{} {}", paint("Saved metric history:", Tone::Success), path);
    }

    if let Some(path) = model_out {
        network.save(path)?;
        println!(
            "{} {}",
            paint("Model saved:", Tone::Success),
            path.display()
        );
    }

    monitor.keep_open_until_closed();
    Ok(network)
}

#[cfg(test)]
mod tests {
    use super::{build_dataset, train_from_dataset};
    use crate::app::types::MonitorOptions;
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
        let result = train_from_dataset(&dataset, &config, false, &MonitorOptions::default(), None);
        assert!(result.is_ok(), "training failed: {:?}", result.err());
    }

    #[test]
    fn train_from_dataset_rejects_dataset_with_fewer_than_three_rows() {
        // Build a minimal Dataset with only 2 rows: col 0 = label, cols 1..31 = features.
        let features = Array2::from_shape_fn((2, 31), |(i, j)| (i + j) as f64 * 0.1);
        let dataset = Dataset {
            features,
            labels: Array1::zeros(2),
            feature_names: Vec::new(),
        };
        let config = one_epoch_config();
        let result = train_from_dataset(&dataset, &config, false, &MonitorOptions::default(), None);
        let Err(e) = result else {
            panic!("expected Err for < 3 rows")
        };
        assert!(e.to_string().contains("at least 3 rows"), "unexpected: {e}");
    }

    #[test]
    fn train_from_dataset_rejects_dataset_with_bad_split_ratios() {
        // n=3: train_end = round(0.85*3=2.55)=3, val_end = 3
        // val_end (3) == train_end (3) → error.
        let features = Array2::from_shape_fn((3, 31), |(i, j)| (i + j) as f64 * 0.1);
        let dataset = Dataset {
            features,
            labels: Array1::zeros(3),
            feature_names: Vec::new(),
        };
        let config = one_epoch_config();
        let result = train_from_dataset(&dataset, &config, false, &MonitorOptions::default(), None);
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
        let result = train_from_dataset(&dataset, &config, false, &opts, None);
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
            false,
            &MonitorOptions::default(),
            Some(Path::new(&model_path)),
        );
        let saved = result.is_ok() && Path::new(&model_path).exists();
        let _ = std::fs::remove_file(&model_path);
        assert!(saved, "model should have been saved before the GUI blocks");
    }
}
