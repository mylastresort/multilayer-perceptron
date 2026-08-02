use mlp::training::monitor::{MonitorMode, MonitoredMetric};

pub enum Subcommand {
    Split,
    Train,
    Predict,
}

pub struct CliArgs {
    pub subcommand: Subcommand,
    pub dataset_path: String,
    pub config_path: String,
    pub verbose: bool,
    #[allow(dead_code)]
    pub gui: bool,
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
            early_stopping: true,
            early_stop_metric: MonitoredMetric::Loss,
            early_stop_mode: MonitorMode::Min,
            early_stop_patience: 60,
            early_stop_min_delta: 0.0,
            early_stop_start_epoch: 0,
            history_out: None,
            metrics: Vec::new(),
        }
    }
}

#[derive(Default)]
pub struct NetOverrides {
    pub learning_rate: Option<f64>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::MonitorOptions;
    use mlp::training::monitor::{MonitorMode, MonitoredMetric};

    #[test]
    fn monitor_options_default_has_sensible_values() {
        let opts = MonitorOptions::default();
        assert!(opts.early_stopping);
        assert_eq!(opts.early_stop_patience, 60);
        assert!(matches!(opts.early_stop_mode, MonitorMode::Min));
        assert!(matches!(opts.early_stop_metric, MonitoredMetric::Loss));
        assert!(opts.history_out.is_none());
    }
}
