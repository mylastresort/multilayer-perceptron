use mlp::training::monitor::{MonitorMode, MonitoredMetric};

pub struct CliArgs {
    pub dataset_path: String,
    pub config_path: String,
    pub verbose: bool,
    pub gui: bool,
    pub monitor_options: MonitorOptions,
    pub net_overrides: NetOverrides,
}

pub struct MonitorOptions {
    pub metrics: Vec<MonitoredMetric>,
    pub early_stopping: bool,
    pub monitor_metric: MonitoredMetric,
    pub monitor_mode: MonitorMode,
    pub monitor_patience: usize,
    pub monitor_min_delta: f64,
    pub monitor_start_epoch: usize,
    pub history_out: Option<String>,
}

impl Default for MonitorOptions {
    fn default() -> Self {
        Self {
            metrics: vec![MonitoredMetric::Loss, MonitoredMetric::Accuracy],
            early_stopping: false,
            monitor_metric: MonitoredMetric::Loss,
            monitor_mode: MonitorMode::Min,
            monitor_patience: 10,
            monitor_min_delta: 0.0,
            monitor_start_epoch: 0,
            history_out: None,
        }
    }
}

#[derive(Default)]
pub struct NetOverrides {
    pub learning_rate: Option<f64>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
}
