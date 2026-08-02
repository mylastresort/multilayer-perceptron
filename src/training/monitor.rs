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
    pub enabled: bool,
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
            enabled: false,
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
        if !self.cfg.enabled || !self.cfg.restore_best_weights {
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
        if !self.cfg.enabled || epoch < self.cfg.start_epoch || logs.is_none() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::callbacks::CallbackLogs;

    #[test]
    fn monitored_metric_parse_all_known_variants() {
        assert_eq!(MonitoredMetric::parse("loss"), Some(MonitoredMetric::Loss));
        assert_eq!(
            MonitoredMetric::parse("accuracy"),
            Some(MonitoredMetric::Accuracy)
        );
        assert_eq!(
            MonitoredMetric::parse("acc"),
            Some(MonitoredMetric::Accuracy)
        );
        assert_eq!(
            MonitoredMetric::parse("precision"),
            Some(MonitoredMetric::Precision)
        );
    }

    #[test]
    fn monitored_metric_parse_unknown_returns_none() {
        assert_eq!(MonitoredMetric::parse("unknown_metric"), None);
        assert_eq!(MonitoredMetric::parse(""), None);
    }

    #[test]
    fn monitored_metric_parse_is_case_insensitive() {
        assert_eq!(MonitoredMetric::parse("LOSS"), Some(MonitoredMetric::Loss));
        assert_eq!(
            MonitoredMetric::parse("Accuracy"),
            Some(MonitoredMetric::Accuracy)
        );
    }

    #[test]
    fn monitored_metric_parse_strips_whitespace() {
        assert_eq!(
            MonitoredMetric::parse("  loss  "),
            Some(MonitoredMetric::Loss)
        );
    }

    #[test]
    fn monitored_metric_as_str_round_trips() {
        let variants = [
            MonitoredMetric::Loss,
            MonitoredMetric::Accuracy,
            MonitoredMetric::Precision,
        ];
        for v in variants {
            let s = v.as_str();
            assert!(
                MonitoredMetric::parse(s).is_some(),
                "as_str round-trip failed for {s}"
            );
        }
    }

    #[test]
    fn monitored_metric_train_value_returns_correct_field() {
        let logs = CallbackLogs {
            loss: Some(0.5),
            accuracy: Some(0.8),
            precision: Some(0.7),
            ..CallbackLogs::default()
        };
        assert_eq!(MonitoredMetric::Loss.train_value(&logs), Some(0.5));
        assert_eq!(MonitoredMetric::Accuracy.train_value(&logs), Some(0.8));
        assert_eq!(MonitoredMetric::Precision.train_value(&logs), Some(0.7));
    }

    #[test]
    fn monitored_metric_val_value_returns_correct_field() {
        let logs = CallbackLogs {
            val_loss: Some(0.4),
            val_accuracy: Some(0.9),
            val_precision: Some(0.85),
            ..CallbackLogs::default()
        };
        assert_eq!(MonitoredMetric::Loss.val_value(&logs), Some(0.4));
        assert_eq!(MonitoredMetric::Accuracy.val_value(&logs), Some(0.9));
        assert_eq!(MonitoredMetric::Precision.val_value(&logs), Some(0.85));
    }

    #[test]
    fn monitor_mode_parse_known_variants() {
        assert!(matches!(MonitorMode::parse("min"), Some(MonitorMode::Min)));
        assert!(matches!(MonitorMode::parse("max"), Some(MonitorMode::Max)));
        assert!(matches!(MonitorMode::parse("MIN"), Some(MonitorMode::Min)));
    }

    #[test]
    fn monitor_mode_parse_unknown_returns_none() {
        assert!(MonitorMode::parse("median").is_none());
    }

    #[test]
    fn history_callback_records_epoch_entries() {
        let mut cb = HistoryCallback::new();
        let logs = CallbackLogs {
            loss: Some(0.5),
            accuracy: Some(0.7),
            ..CallbackLogs::default()
        };
        cb.on_epoch_end(0, Some(&logs));
        cb.on_epoch_end(1, Some(&logs));

        assert_eq!(cb.history().epochs.len(), 2);
        assert_eq!(cb.history().epochs[0].epoch, 0);
        assert_eq!(cb.history().epochs[0].loss, Some(0.5));
        assert_eq!(cb.history().epochs[1].epoch, 1);
    }

    #[test]
    fn history_callback_ignores_epoch_end_with_no_logs() {
        let mut cb = HistoryCallback::new();
        cb.on_epoch_end(0, None);
        assert_eq!(cb.history().epochs.len(), 0);
    }

    #[test]
    fn history_callback_save_json_writes_readable_file() {
        let mut cb = HistoryCallback::new();
        let logs = CallbackLogs {
            loss: Some(0.3),
            ..CallbackLogs::default()
        };
        cb.on_epoch_end(0, Some(&logs));

        let path =
            std::env::temp_dir().join(format!("mlp_history_test_{}.json", std::process::id()));
        cb.save_json(path.to_str().unwrap()).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(contents.contains("epochs"));
    }

    fn make_cfg(mode: MonitorMode, patience: usize) -> EarlyStoppingConfig {
        EarlyStoppingConfig {
            enabled: true,
            metric: MonitoredMetric::Loss,
            mode,
            patience,
            min_delta: 0.0,
            start_epoch: 0,
            restore_best_weights: true,
        }
    }

    fn test_network() -> crate::network::model::Network {
        use crate::network::{
            activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
        };
        Network::builder()
            .add_layer(Layer::new(
                2,
                3,
                ActivationFunction::Sigmoid,
                WeightInitializer::Xavier,
            ))
            .add_layer(Layer::new(
                3,
                2,
                ActivationFunction::Sigmoid,
                WeightInitializer::Xavier,
            ))
            .add_layer(Layer::new(
                2,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::Xavier,
            ))
            .build()
    }

    fn loss_logs(loss: f64) -> CallbackLogs {
        CallbackLogs {
            loss: Some(loss),
            ..CallbackLogs::default()
        }
    }

    #[test]
    fn early_stopping_does_not_stop_when_loss_keeps_improving() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 2));
        for epoch in 0..5 {
            let l = 1.0 - epoch as f64 * 0.1;
            cb.on_epoch_end(epoch, Some(&loss_logs(l)));
        }
        assert!(!cb.stopped());
        assert!(!cb.should_stop());
    }

    #[test]
    fn early_stopping_stops_after_patience_consecutive_epochs() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 2));
        cb.on_epoch_end(0, Some(&loss_logs(1.0)));
        cb.on_epoch_end(1, Some(&loss_logs(1.0)));
        assert!(!cb.stopped());
        cb.on_epoch_end(2, Some(&loss_logs(1.0)));
        assert!(cb.stopped());
    }

    #[test]
    fn early_stopping_respects_start_epoch() {
        let mut cfg = make_cfg(MonitorMode::Min, 1);
        cfg.start_epoch = 5;
        let mut cb = EarlyStoppingCallback::new(cfg);

        for epoch in 0..5 {
            cb.on_epoch_end(epoch, Some(&loss_logs(2.0)));
        }
        assert!(!cb.stopped());
    }

    #[test]
    fn early_stopping_respects_max_mode() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Max, 1));

        let logs_acc = |v: f64| CallbackLogs {
            loss: Some(v),
            ..CallbackLogs::default()
        };
        cb.on_epoch_end(0, Some(&logs_acc(0.5)));
        cb.on_epoch_end(1, Some(&logs_acc(0.6)));
        cb.on_epoch_end(2, Some(&logs_acc(0.7)));
        assert!(!cb.stopped());
    }

    #[test]
    fn early_stopping_disabled_never_stops() {
        let cfg = EarlyStoppingConfig {
            enabled: false,
            ..EarlyStoppingConfig::default()
        };
        let mut cb = EarlyStoppingCallback::new(cfg);
        for epoch in 0..10 {
            cb.on_epoch_end(epoch, Some(&loss_logs(1.0)));
        }
        assert!(!cb.stopped());
    }

    #[test]
    fn early_stopping_records_stopped_epoch() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 2));
        cb.on_epoch_end(0, Some(&loss_logs(1.0)));
        cb.on_epoch_end(1, Some(&loss_logs(1.0)));
        assert_eq!(cb.stopped_epoch(), None);
        cb.on_epoch_end(2, Some(&loss_logs(1.0)));
        assert_eq!(cb.stopped_epoch(), Some(2));
    }

    #[test]
    fn early_stopping_restores_best_weights() {
        let mut net = test_network();

        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 1));

        cb.on_epoch_end(0, Some(&loss_logs(1.0)));
        cb.on_epoch_end_network(0, Some(&loss_logs(1.0)), &mut net);
        let best: Vec<Array2<f64>> = net.layers.iter().map(|l| l.weights.clone()).collect();

        cb.on_epoch_end(1, Some(&loss_logs(2.0)));
        cb.on_epoch_end_network(1, Some(&loss_logs(2.0)), &mut net);
        assert!(cb.stopped());
        assert_eq!(cb.stopped_epoch(), Some(1));

        for layer in net.layers.iter_mut() {
            layer.weights = layer.weights.clone() + 100.0;
        }
        cb.restore_best(&mut net);

        for (i, layer) in net.layers.iter().enumerate() {
            for ((r, c), v) in layer.weights.indexed_iter() {
                assert!(
                    (v - best[i][[r, c]]).abs() < 1e-12,
                    "layer {i} weights not restored at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn early_stopping_disabled_is_fully_inert() {
        let mut net = test_network();
        let original: Vec<Array2<f64>> = net.layers.iter().map(|l| l.weights.clone()).collect();

        let mut cfg = make_cfg(MonitorMode::Min, 1);
        cfg.enabled = false;
        let mut cb = EarlyStoppingCallback::new(cfg);

        for epoch in 0..5 {
            let l = 1.0 - epoch as f64 * 0.1;
            cb.on_epoch_end(epoch, Some(&loss_logs(l)));
            cb.on_epoch_end_network(epoch, Some(&loss_logs(l)), &mut net);
        }

        assert!(!cb.stopped());
        assert_eq!(cb.stopped_epoch(), None);

        for layer in net.layers.iter_mut() {
            layer.weights = layer.weights.clone() + 100.0;
        }
        cb.restore_best(&mut net);
        for (i, layer) in net.layers.iter().enumerate() {
            for ((r, c), v) in layer.weights.indexed_iter() {
                assert!(
                    (v - (original[i][[r, c]] + 100.0)).abs() < 1e-12,
                    "restore_best mutated weights while disabled at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn early_stopping_does_not_restore_when_restore_best_weights_false() {
        let mut net = test_network();
        let original: Vec<Array2<f64>> = net.layers.iter().map(|l| l.weights.clone()).collect();

        let mut cfg = make_cfg(MonitorMode::Min, 5);
        cfg.restore_best_weights = false;
        let mut cb = EarlyStoppingCallback::new(cfg);

        cb.on_epoch_end(0, Some(&loss_logs(1.0)));
        cb.on_epoch_end_network(0, Some(&loss_logs(1.0)), &mut net);

        for layer in net.layers.iter_mut() {
            layer.weights = layer.weights.clone() + 100.0;
        }
        cb.restore_best(&mut net);
        for (i, layer) in net.layers.iter().enumerate() {
            for ((r, c), v) in layer.weights.indexed_iter() {
                assert!(
                    (v - (original[i][[r, c]] + 100.0)).abs() < 1e-12,
                    "restore_best ran despite restore_best_weights=false at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn early_stopping_ignores_epoch_end_with_no_logs() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 1));
        cb.on_epoch_end(0, None);
        assert!(!cb.stopped());
        assert_eq!(cb.stopped_epoch(), None);
    }

    #[test]
    fn early_stopping_returns_early_when_monitored_metric_absent_from_logs() {
        let mut cfg = make_cfg(MonitorMode::Min, 1);
        cfg.metric = MonitoredMetric::Accuracy;
        let mut cb = EarlyStoppingCallback::new(cfg);
        let logs = CallbackLogs {
            loss: Some(0.5),
            ..CallbackLogs::default()
        };
        cb.on_epoch_end(0, Some(&logs));
        assert!(!cb.stopped());
        assert_eq!(cb.stopped_epoch(), None);
    }
}
