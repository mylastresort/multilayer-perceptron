use std::{error::Error, fs};

use serde::Serialize;

use crate::network::callbacks::{Callback, CallbackLogs};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitoredMetric {
    Loss,
    Accuracy,
    Precision,
    Recall,
    F1,
}

impl MonitoredMetric {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "loss" => Some(Self::Loss),
            "accuracy" | "acc" => Some(Self::Accuracy),
            "precision" => Some(Self::Precision),
            "recall" => Some(Self::Recall),
            "f1" | "f1_score" => Some(Self::F1),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loss => "loss",
            Self::Accuracy => "accuracy",
            Self::Precision => "precision",
            Self::Recall => "recall",
            Self::F1 => "f1",
        }
    }

    pub fn train_value(self, logs: &CallbackLogs) -> Option<f64> {
        match self {
            Self::Loss => logs.loss,
            Self::Accuracy => logs.accuracy,
            Self::Precision => logs.precision,
            Self::Recall => logs.recall,
            Self::F1 => logs.f1,
        }
    }

    pub fn val_value(self, logs: &CallbackLogs) -> Option<f64> {
        match self {
            Self::Loss => logs.val_loss,
            Self::Accuracy => logs.val_accuracy,
            Self::Precision => logs.val_precision,
            Self::Recall => logs.val_recall,
            Self::F1 => logs.val_f1,
        }
    }
}

pub fn parse_monitored_metrics(csv: &str) -> Result<Vec<MonitoredMetric>, Box<dyn Error>> {
    let mut metrics = Vec::new();
    for token in csv.split(',') {
        let Some(metric) = MonitoredMetric::parse(token) else {
            return Err(format!("Unknown monitored metric: {}", token.trim()).into());
        };
        if !metrics.contains(&metric) {
            metrics.push(metric);
        }
    }

    if metrics.is_empty() {
        return Err("At least one metric is required for --monitor-metrics".into());
    }

    Ok(metrics)
}

#[derive(Debug, Clone, Copy)]
pub enum MonitorMode {
    Min,
    Max,
}

impl MonitorMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochHistoryEntry {
    pub epoch: usize,
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub val_precision: Option<f64>,
    pub recall: Option<f64>,
    pub val_recall: Option<f64>,
    pub f1: Option<f64>,
    pub val_f1: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TrainingHistory {
    pub epochs: Vec<EpochHistoryEntry>,
}

pub struct HistoryCallback {
    history: TrainingHistory,
}

impl HistoryCallback {
    pub fn new() -> Self {
        Self {
            history: TrainingHistory::default(),
        }
    }

    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(&self.history)?;
        fs::write(path, json)?;
        Ok(())
    }
}

impl Callback for HistoryCallback {
    fn on_epoch_end(&mut self, epoch: usize, logs: Option<&CallbackLogs>) {
        let Some(logs) = logs else {
            return;
        };

        self.history.epochs.push(EpochHistoryEntry {
            epoch,
            loss: logs.loss,
            val_loss: logs.val_loss,
            accuracy: logs.accuracy,
            val_accuracy: logs.val_accuracy,
            precision: logs.precision,
            val_precision: logs.val_precision,
            recall: logs.recall,
            val_recall: logs.val_recall,
            f1: logs.f1,
            val_f1: logs.val_f1,
        });
    }
}

#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub metric: MonitoredMetric,
    pub mode: MonitorMode,
    pub patience: usize,
    pub min_delta: f64,
    pub start_epoch: usize,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            metric: MonitoredMetric::Loss,
            mode: MonitorMode::Min,
            patience: 10,
            min_delta: 0.0,
            start_epoch: 0,
        }
    }
}

pub struct EarlyStoppingCallback {
    cfg: EarlyStoppingConfig,
    best_value: Option<f64>,
    best_epoch: Option<usize>,
    wait: usize,
    stop: bool,
}

impl EarlyStoppingCallback {
    pub fn new(cfg: EarlyStoppingConfig) -> Self {
        Self {
            cfg,
            best_value: None,
            best_epoch: None,
            wait: 0,
            stop: false,
        }
    }

    pub fn stopped(&self) -> bool {
        self.stop
    }

    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    fn improved(&self, current: f64, best: f64) -> bool {
        match self.cfg.mode {
            MonitorMode::Min => current < best - self.cfg.min_delta,
            MonitorMode::Max => current > best + self.cfg.min_delta,
        }
    }
}

impl Callback for EarlyStoppingCallback {
    fn on_epoch_end(&mut self, epoch: usize, logs: Option<&CallbackLogs>) {
        if !self.cfg.enabled || epoch < self.cfg.start_epoch {
            return;
        }

        let Some(logs) = logs else {
            return;
        };

        let current = self
            .cfg
            .metric
            .val_value(logs)
            .or_else(|| self.cfg.metric.train_value(logs));
        let Some(current) = current else {
            return;
        };

        match self.best_value {
            None => {
                self.best_value = Some(current);
                self.best_epoch = Some(epoch);
                self.wait = 0;
            }
            Some(best) if self.improved(current, best) => {
                self.best_value = Some(current);
                self.best_epoch = Some(epoch);
                self.wait = 0;
            }
            Some(_) => {
                self.wait += 1;
                if self.wait > self.cfg.patience {
                    self.stop = true;
                }
            }
        }
    }

    fn should_stop(&self) -> bool {
        self.stop
    }
}
