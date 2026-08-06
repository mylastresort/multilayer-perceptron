use std::{error::Error, fs};

use ndarray::{Array1, Array2};
use serde::Serialize;

use crate::network::callbacks::{Callback, CallbackLogs};
use crate::network::model::Network;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitoredMetric {
    Loss,
    Accuracy,
    Precision,
}

pub fn parse_monitored_metrics(value: &str) -> Result<Vec<MonitoredMetric>, String> {
    value
        .split(',')
        .map(|part| {
            MonitoredMetric::parse(part)
                .ok_or_else(|| format!("unknown metric '{}' in --monitor-metrics", part.trim()))
        })
        .collect()
}

impl MonitoredMetric {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "loss" => Some(Self::Loss),
            "accuracy" | "acc" => Some(Self::Accuracy),
            "precision" => Some(Self::Precision),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loss => "loss",
            Self::Accuracy => "accuracy",
            Self::Precision => "precision",
        }
    }

    pub fn train_value(self, logs: &CallbackLogs) -> Option<f64> {
        match self {
            Self::Loss => logs.loss,
            Self::Accuracy => logs.accuracy,
            Self::Precision => logs.precision,
        }
    }

    pub fn val_value(self, logs: &CallbackLogs) -> Option<f64> {
        match self {
            Self::Loss => logs.val_loss,
            Self::Accuracy => logs.val_accuracy,
            Self::Precision => logs.val_precision,
        }
    }
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

impl Default for HistoryCallback {
    fn default() -> Self {
        Self::new()
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
        });
    }
}

#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub metric: MonitoredMetric,
    pub mode: MonitorMode,
    pub patience: usize,
    pub min_delta: f64,
    pub start_epoch: usize,
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            metric: MonitoredMetric::Loss,
            mode: MonitorMode::Min,
            patience: 10,
            min_delta: 0.0,
            start_epoch: 0,
            restore_best_weights: true,
        }
    }
}

pub struct EarlyStoppingCallback {
    cfg: EarlyStoppingConfig,
    best_value: Option<f64>,
    best_weights: Option<Vec<(Array2<f64>, Array1<f64>)>>,
    improved_this_epoch: bool,
    wait: usize,
    stopped_epoch: Option<usize>,
    stop: bool,
}

impl EarlyStoppingCallback {
    pub fn new(cfg: EarlyStoppingConfig) -> Self {
        Self {
            cfg,
            best_value: None,
            best_weights: None,
            improved_this_epoch: false,
            wait: 0,
            stopped_epoch: None,
            stop: false,
        }
    }

    pub fn stopped(&self) -> bool {
        self.stop
    }

    pub fn stopped_epoch(&self) -> Option<usize> {
        self.stopped_epoch
    }

    pub fn restore_best(&self, network: &mut Network) {
        if !self.cfg.restore_best_weights {
            return;
        }
        if let Some(best) = &self.best_weights {
            for (layer, (w, b)) in network.layers.iter_mut().zip(best.iter()) {
                layer.weights = w.clone();
                layer.bias = b.clone();
            }
        }
    }

    fn improved(&self, current: f64, best: f64) -> bool {
        match self.cfg.mode {
            MonitorMode::Min => current < best - self.cfg.min_delta,
            MonitorMode::Max => current > best + self.cfg.min_delta,
        }
    }

    fn update_best(&mut self, current: f64) {
        self.wait += 1;
        match self.best_value {
            None => {
                self.best_value = Some(current);
                self.improved_this_epoch = true;
                self.wait = 0;
            }
            Some(best) if self.improved(current, best) => {
                self.best_value = Some(current);
                self.improved_this_epoch = true;
                self.wait = 0;
            }
            Some(_) => {
                self.improved_this_epoch = false;
            }
        }
    }
}

impl Callback for EarlyStoppingCallback {
    fn on_epoch_end(&mut self, epoch: usize, logs: Option<&CallbackLogs>) {
        if epoch < self.cfg.start_epoch {
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

        self.update_best(current);

        if self.wait >= self.cfg.patience {
            self.stopped_epoch = Some(epoch);
            self.stop = true;
        }
    }

    fn on_epoch_end_network(
        &mut self,
        epoch: usize,
        logs: Option<&CallbackLogs>,
        network: &mut Network,
    ) {
        if epoch < self.cfg.start_epoch || logs.is_none() {
            return;
        }
        if !self.cfg.restore_best_weights {
            return;
        }

        if self.improved_this_epoch {
            self.best_weights = Some(
                network
                    .layers
                    .iter()
                    .map(|l| (l.weights.clone(), l.bias.clone()))
                    .collect(),
            );
            self.improved_this_epoch = false;
        } else if self.best_weights.is_none() && self.best_value.is_some() {
            self.best_weights = Some(
                network
                    .layers
                    .iter()
                    .map(|l| (l.weights.clone(), l.bias.clone()))
                    .collect(),
            );
        }
    }

    fn should_stop(&self) -> bool {
        self.stop
    }
}
