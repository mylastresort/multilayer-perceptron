use std::error::Error;
use std::path::Path;

use mlp::console::{Tone, bold, paint};
use mlp::data::loader::{Dataset, load_dataset};
use mlp::network::callbacks::{Callback, ProgressLogger};
use mlp::network::config::NetworkConfig;
use mlp::network::model::Network;
use mlp::training::metrics::Metrics;
use mlp::training::monitor::{
    EarlyStoppingCallback, EarlyStoppingConfig, HistoryCallback, MonitoredMetric,
};
use mlp::training::optimizer::OptimizerType;
use mlp::visualization::plotter::{TrainingHistory as PlotTrainingHistory, plot_training_curves};
use ndarray::{Array1, Array2, Axis, s};

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

    load_dataset(dataset_path, 1, names, 0)
}

struct PreparedData {
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    x_val: Array2<f64>,
    y_val: Array1<f64>,
    train_mean: Array1<f64>,
    train_std: Array1<f64>,
}

fn training_split_index(n: usize) -> Result<usize, Box<dyn Error>> {
    if n < 3 {
        return Err("dataset must contain at least 3 rows to split train/val/test".into());
    }

    let train_end = (0.85 * n as f64).round() as usize;
    if train_end == 0 || n <= train_end {
        return Err("dataset split produced empty training or validation set".into());
    }
    Ok(train_end)
}

fn prepare_training_data(dataset: &Dataset) -> Result<PreparedData, Box<dyn Error>> {
    let x_raw = dataset.features.slice(s![.., 1..]).to_owned();
    let y = dataset
        .features
        .column(0)
        .mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });

    let train_end = training_split_index(x_raw.nrows())?;
    let n = x_raw.nrows();

    let x_train_raw = x_raw.slice(s![0..train_end, ..]).to_owned();
    let x_val_raw = x_raw.slice(s![train_end..n, ..]).to_owned();
    let y_train = y.slice(s![0..train_end]).to_owned();
    let y_val = y.slice(s![train_end..n]).to_owned();

    let train_mean = x_train_raw
        .mean_axis(Axis(0))
        .expect("training features should not be empty");
    let train_std = x_train_raw.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));

    let x_train = (&x_train_raw - &train_mean) / &train_std;
    let x_val = (&x_val_raw - &train_mean) / &train_std;

    Ok(PreparedData {
        x_train,
        y_train,
        x_val,
        y_val,
        train_mean,
        train_std,
    })
}

fn timestamp_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let yy = if m <= 2 { y + 1 } else { y };
    let h = sod / 3600;
    let mi = (sod % 3600) / 60;
    let s = sod % 60;
    format!("{yy:04}{m:02}{d:02}-{h:02}{mi:02}{s:02}")
}

fn plot_history_from(history: &HistoryCallback) -> PlotTrainingHistory {
    let epochs = &history.history().epochs;
    PlotTrainingHistory {
        train_loss: epochs.iter().filter_map(|e| e.loss).collect(),
        val_loss: epochs.iter().filter_map(|e| e.val_loss).collect(),
        train_accuracy: epochs.iter().filter_map(|e| e.accuracy).collect(),
        val_accuracy: epochs.iter().filter_map(|e| e.val_accuracy).collect(),
        train_precision: epochs.iter().filter_map(|e| e.precision).collect(),
        val_precision: epochs.iter().filter_map(|e| e.val_precision).collect(),
    }
}

fn print_summary_line(metrics: &Metrics) {
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
}

fn print_training_summary(
    metrics: &Metrics,
    monitor_options: &MonitorOptions,
    early_stopping: &EarlyStoppingCallback,
    history: &HistoryCallback,
) {
    print_summary_line(metrics);
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
                &format!(
                    "(monitored: {})",
                    monitor_options.early_stop_metric.as_str()
                ),
                Tone::Warn
            )
        );
    }
}

fn curve_file_path(config_name: Option<&str>) -> String {
    let config_tag = config_name
        .map(|name| format!("{name}_"))
        .unwrap_or_default();
    format!(
        "reports/learning_curves_{config_tag}{}.png",
        timestamp_utc()
    )
}

fn ensure_reports_dir() -> bool {
    if let Err(e) = std::fs::create_dir_all("reports") {
        eprintln!(
            "{} {e}",
            paint(
                "Warning: could not create reports/ for learning curves:",
                Tone::Warn
            )
        );
        return false;
    }
    true
}

fn export_learning_curves(
    history: &HistoryCallback,
    config_name: Option<&str>,
    metrics: &[MonitoredMetric],
) {
    if !ensure_reports_dir() {
        return;
    }
    let curve_path = curve_file_path(config_name);
    let plot_history = plot_history_from(history);
    match plot_training_curves(&plot_history, &curve_path, metrics) {
        Ok(()) => println!(
            "{} {}",
            paint("Learning curves saved:", Tone::Success),
            curve_path
        ),
        Err(e) => eprintln!(
            "{} {e}",
            paint("Warning: could not save learning curves:", Tone::Warn)
        ),
    }
}

fn setup_callbacks(
    monitor_options: &MonitorOptions,
    epochs: usize,
) -> (HistoryCallback, EarlyStoppingCallback, ProgressLogger) {
    let history = HistoryCallback::new();
    let early_stopping = EarlyStoppingCallback::new(EarlyStoppingConfig {
        enabled: monitor_options.early_stopping,
        metric: monitor_options.early_stop_metric,
        mode: monitor_options.early_stop_mode,
        patience: monitor_options.early_stop_patience,
        min_delta: monitor_options.early_stop_min_delta,
        start_epoch: monitor_options.early_stop_start_epoch,
        restore_best_weights: true,
    });
    let progress_logger = ProgressLogger::new(epochs);
    (history, early_stopping, progress_logger)
}

fn fit_network(
    network: &mut Network,
    data: &PreparedData,
    network_config: &NetworkConfig,
    callbacks: &mut [&mut dyn Callback],
) -> Metrics {
    network.fit_with_callbacks(
        data.x_train.view(),
        data.y_train.view(),
        Some((data.x_val.view(), data.y_val.view())),
        mlp::network::model::FitConfig {
            batch_size: network_config.batch_size,
            epochs: network_config.epochs,
            optimizer: OptimizerType::for_kind(
                network_config.optimizer,
                network_config.weight_decay,
            ),
            loss_fn: network_config.loss,
        },
        callbacks,
    )
}

fn save_artifacts(
    network: &mut Network,
    history: &HistoryCallback,
    monitor_options: &MonitorOptions,
    model_out: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

pub fn train_from_dataset(
    dataset: &Dataset,
    network_config: &NetworkConfig,
    monitor_options: &MonitorOptions,
    config_name: Option<&str>,
    model_out: Option<&Path>,
) -> Result<Network, Box<dyn Error>> {
    let data = prepare_training_data(dataset)?;

    let mut network = network_config.build_network();
    network.feature_mean = Some(data.train_mean.clone());
    network.feature_std = Some(data.train_std.clone());

    let (mut history, mut early_stopping, mut progress_logger) =
        setup_callbacks(monitor_options, network_config.epochs);
    let mut callbacks: Vec<&mut dyn Callback> =
        vec![&mut history, &mut early_stopping, &mut progress_logger];

    let metrics = fit_network(&mut network, &data, network_config, &mut callbacks);
    drop(callbacks);

    early_stopping.restore_best(&mut network);

    print_training_summary(&metrics, monitor_options, &early_stopping, &history);
    save_artifacts(&mut network, &history, monitor_options, model_out)?;

    export_learning_curves(&history, config_name, &monitor_options.metrics);

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
        let result = train_from_dataset(&dataset, &config, &MonitorOptions::default(), None, None);
        assert!(result.is_ok(), "training failed: {:?}", result.err());
    }

    #[test]
    fn train_from_dataset_rejects_dataset_with_fewer_than_three_rows() {
        let features = Array2::from_shape_fn((2, 31), |(i, j)| (i + j) as f64 * 0.1);
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
        let features = Array2::from_shape_fn((3, 31), |(i, j)| (i + j) as f64 * 0.1);
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
}
