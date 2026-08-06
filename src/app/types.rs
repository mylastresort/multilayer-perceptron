use crate::training::monitor::{MonitorMode, MonitoredMetric};

pub enum Subcommand {
    Split,
    Train,
    Predict,
}

pub struct CliArgs {
    pub subcommand: Subcommand,
    pub dataset_path: String,
    pub config_path: String,
    pub monitor_options: MonitorOptions,
    pub net_overrides: NetOverrides,
    pub model_out: Option<String>,
    pub model_in: Option<String>,
    pub split_ratio: f64,
    pub train_out: Option<String>,
    pub val_out: Option<String>,
}

pub struct MonitorOptions {
    pub early_stopping: bool,
    pub early_stop_metric: MonitoredMetric,
    pub early_stop_mode: MonitorMode,
    pub early_stop_patience: usize,
    pub early_stop_min_delta: f64,
    pub early_stop_start_epoch: usize,
    pub history_out: Option<String>,
    pub metrics: Vec<MonitoredMetric>,
}

impl Default for MonitorOptions {
    fn default() -> Self {
        Self {
            early_stopping: false,
            early_stop_metric: MonitoredMetric::Loss,
            early_stop_mode: MonitorMode::Min,
            early_stop_patience: 60,
            early_stop_min_delta: 0.0,
            early_stop_start_epoch: 0,
            history_out: None,
            metrics: vec![
                MonitoredMetric::Loss,
                MonitoredMetric::Accuracy,
                MonitoredMetric::Precision,
            ],
        }
    }
}

#[derive(Default)]
pub struct NetOverrides {
    pub learning_rate: Option<f64>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
}
