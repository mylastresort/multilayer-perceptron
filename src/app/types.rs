use mlp::training::monitor::{MonitorMode, MonitoredMetric};

/// Top-level command selected via the first positional argument.
pub enum Subcommand {
    /// Split the dataset into train/validation CSV files.
    Split,
    /// Train the network and save the model to a file.
    Train,
    /// Load a saved model and evaluate it with binary cross-entropy.
    Predict,
}

pub struct CliArgs {
    pub subcommand: Subcommand,
    pub dataset_path: String,
    pub config_path: String,
    pub verbose: bool,
    /// Open the live learning-curve GUI (`--gui`).
    #[allow(dead_code)] // advertised flag; GUI not yet implemented
    pub gui: bool,
    pub monitor_options: MonitorOptions,
    pub net_overrides: NetOverrides,
    /// Output path for the saved model (used by `train`).
    pub model_out: Option<String>,
    /// Path to a previously saved model (used by `predict`).
    pub model_in: Option<String>,
    /// Fraction of the dataset used as the training split (used by `split`).
    pub split_ratio: f64,
    /// Output CSV path for the training split (used by `split`).
    pub train_out: Option<String>,
    /// Output CSV path for the validation split (used by `split`).
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
    /// Metrics selected via `--monitor-metrics` for history/plotting.
    pub metrics: Vec<MonitoredMetric>,
}

impl Default for MonitorOptions {
    fn default() -> Self {
        Self {
            // Early stopping is on by default so `restore_best_weights` keeps the
            // best-epoch model; --no-early-stopping still opts out.
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
