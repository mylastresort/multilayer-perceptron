use std::error::Error;
use std::path::Path;

use crate::console::{Tone, bold, paint};
use crate::data::loader::{Dataset, load_dataset};
use crate::data::preprocessing::{Normalizer, StandardScaler};
use crate::data::split::stratified_split_by_target;
use crate::network::callbacks::{Callback, ProgressLogger};
use crate::network::config::NetworkConfig;
use crate::network::model::Network;
use crate::network::model::FitConfig;
use crate::training::metrics::Metrics;
use crate::training::monitor::{
    EarlyStoppingCallback, EarlyStoppingConfig, HistoryCallback, MonitoredMetric,
};
use crate::training::optimizer::OptimizerType;
use crate::visualization::plotter::{TrainingHistory as PlotTrainingHistory, plot_training_curves};
use ndarray::{Array1, Array2, s};

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
    scaler: StandardScaler,
}

pub(crate) struct PreparedFeatures {
    pub(crate) x: Array2<f64>,
    pub(crate) scaler: StandardScaler,
}

pub(crate) fn extract_features_target(dataset: &Dataset) -> (Array2<f64>, Array1<f64>) {
    let features = dataset.features.slice(s![.., 1..]).to_owned();
    let target = dataset
        .features
        .column(0)
        .mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
    (features, target)
}

pub(crate) fn prepare_data(
    x_raw: &Array2<f64>,
    scaler: Option<&StandardScaler>,
) -> Result<PreparedFeatures, Box<dyn Error>> {
    if x_raw.nrows() == 0 {
        return Err("dataset has no rows".into());
    }
    match scaler {
        Some(scaler) => Ok(PreparedFeatures {
            x: scaler.transform(x_raw),
            scaler: scaler.clone(),
        }),
        None => {
            let mut scaler = StandardScaler::default();
            let x = scaler.fit_transform(x_raw);
            Ok(PreparedFeatures { x, scaler })
        }
    }
}

fn prepare_training_data(dataset: &Dataset) -> Result<PreparedData, Box<dyn Error>> {
    let (x_raw, y) = extract_features_target(dataset);
    if x_raw.nrows() < 3 {
        return Err("dataset must contain at least 3 rows to split train/val/test".into());
    }

    let (train_ds, val_ds) = stratified_split_by_target(dataset, &y, 0.85, None);
    let (x_train_raw, y_train) = extract_features_target(&train_ds);
    let (x_val_raw, y_val) = extract_features_target(&val_ds);
    if x_train_raw.nrows() == 0 || x_val_raw.nrows() == 0 {
        return Err("dataset split produced empty training or validation set".into());
    }

    let train = prepare_data(&x_train_raw, None)?;
    let val = prepare_data(&x_val_raw, Some(&train.scaler))?;

    Ok(PreparedData {
        x_train: train.x,
        y_train,
        x_val: val.x,
        y_val,
        scaler: train.scaler,
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
    early_stopping: Option<&EarlyStoppingCallback>,
    history: Option<&HistoryCallback>,
) {
    print_summary_line(metrics);
    if let Some(history) = history {
        println!(
            "{} {}",
            paint("History records:", Tone::Info),
            paint(
                &format!("{} epochs", history.history().epochs.len()),
                Tone::Accent
            )
        );
    }
    if let Some(early_stopping) = early_stopping
        && let Some(stopped_epoch) = early_stopping.stopped_epoch()
    {
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
    history: Option<&HistoryCallback>,
    config_name: Option<&str>,
    metrics: &[MonitoredMetric],
) {
    let Some(history) = history else {
        return;
    };
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

struct RegisteredCallbacks {
    history: Option<HistoryCallback>,
    early_stopping: Option<EarlyStoppingCallback>,
    progress_logger: ProgressLogger,
}

impl RegisteredCallbacks {
    fn enabled(&mut self) -> Vec<&mut dyn Callback> {
        let mut callbacks: Vec<&mut dyn Callback> = Vec::new();
        if let Some(history) = self.history.as_mut() {
            callbacks.push(history);
        }
        if let Some(early_stopping) = self.early_stopping.as_mut() {
            callbacks.push(early_stopping);
        }
        callbacks.push(&mut self.progress_logger);
        callbacks
    }
}

fn setup_callbacks(monitor_options: &MonitorOptions, epochs: usize) -> RegisteredCallbacks {
    let history_enabled = monitor_options.history_out.is_some() || !monitor_options.metrics.is_empty();
    RegisteredCallbacks {
        history: history_enabled.then(HistoryCallback::new),
        early_stopping: monitor_options.early_stopping.then(|| {
            EarlyStoppingCallback::new(EarlyStoppingConfig {
                metric: monitor_options.early_stop_metric,
                mode: monitor_options.early_stop_mode,
                patience: monitor_options.early_stop_patience,
                min_delta: monitor_options.early_stop_min_delta,
                start_epoch: monitor_options.early_stop_start_epoch,
                restore_best_weights: true,
            })
        }),
        progress_logger: ProgressLogger::new(epochs),
    }
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
        FitConfig {
            batch_size: network_config.batch_size,
            epochs: network_config.epochs,
            optimizer: OptimizerType::for_kind(network_config.optimizer),
            loss_fn: network_config.loss,
        },
        callbacks,
    )
}

fn save_artifacts(
    network: &mut Network,
    history: Option<&HistoryCallback>,
    monitor_options: &MonitorOptions,
    model_out: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    if let Some(path) = &monitor_options.history_out
        && let Some(history) = history
    {
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
    network.scaler = Some(data.scaler.clone());

    let mut callbacks = setup_callbacks(monitor_options, network_config.epochs);
    let mut enabled: Vec<&mut dyn Callback> = callbacks.enabled();
    let metrics = fit_network(&mut network, &data, network_config, &mut enabled);
    drop(enabled);

    if let Some(early_stopping) = callbacks.early_stopping.as_ref() {
        early_stopping.restore_best(&mut network);
    }

    print_training_summary(
        &metrics,
        monitor_options,
        callbacks.early_stopping.as_ref(),
        callbacks.history.as_ref(),
    );
    save_artifacts(&mut network, callbacks.history.as_ref(), monitor_options, model_out)?;

    export_learning_curves(callbacks.history.as_ref(), config_name, &monitor_options.metrics);

    Ok(network)
}

