use std::error::Error;

use mlp::console::{Tone, bold, paint};
use mlp::data::loader::{Dataset, load_dataset};
use mlp::network::callbacks::{Callback, ProgressLogger};
use mlp::network::config::NetworkConfig;
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
) -> Result<(), Box<dyn Error>> {
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

    let train_end = (0.70 * n as f64).round() as usize;
    let val_end = (0.85 * n as f64).round() as usize;
    if train_end == 0 || val_end <= train_end {
        return Err("dataset split produced empty training or validation set".into());
    }

    let x_train_raw = x_raw.slice(s![0..train_end, ..]).to_owned();
    let x_val_raw = x_raw.slice(s![train_end..val_end, ..]).to_owned();
    let y_train = y.slice(s![0..train_end]).to_owned();
    let y_val = y.slice(s![train_end..val_end]).to_owned();
    let (x_train, x_val) = standardize_from_train(&x_train_raw, &x_val_raw);

    let mut network = network_config.build_network();
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
        OptimizerType::from(network_config.optimizer),
        LossFunction::CategoricalCrossEntropy,
        &mut callbacks,
    );

    drop(callbacks);

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
    if let Some(best_epoch) = early_stopping.best_epoch() {
        println!(
            "{} {}",
            paint("Early stopping best epoch:", Tone::Warn),
            paint(&best_epoch.to_string(), Tone::Warn)
        );
    }
    if early_stopping.stopped() {
        println!(
            "{} {}",
            paint("Early stopping triggered on metric", Tone::Warn),
            paint(monitor_options.monitor_metric.as_str(), Tone::Warn)
        );
    }

    if let Some(path) = &monitor_options.history_out {
        history.save_json(path)?;
        println!("{} {}", paint("Saved metric history:", Tone::Success), path);
    }

    monitor.keep_open_until_closed();
    Ok(())
}
