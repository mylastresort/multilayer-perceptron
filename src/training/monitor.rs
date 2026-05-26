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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::callbacks::CallbackLogs;

    // -----------------------------------------------------------------------
    // MonitoredMetric::parse
    // -----------------------------------------------------------------------

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
        assert_eq!(
            MonitoredMetric::parse("recall"),
            Some(MonitoredMetric::Recall)
        );
        assert_eq!(MonitoredMetric::parse("f1"), Some(MonitoredMetric::F1));
        assert_eq!(
            MonitoredMetric::parse("f1_score"),
            Some(MonitoredMetric::F1)
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

    // -----------------------------------------------------------------------
    // MonitoredMetric::as_str
    // -----------------------------------------------------------------------

    #[test]
    fn monitored_metric_as_str_round_trips() {
        let variants = [
            MonitoredMetric::Loss,
            MonitoredMetric::Accuracy,
            MonitoredMetric::Precision,
            MonitoredMetric::Recall,
            MonitoredMetric::F1,
        ];
        for v in variants {
            let s = v.as_str();
            assert!(
                MonitoredMetric::parse(s).is_some(),
                "as_str round-trip failed for {s}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // MonitoredMetric::train_value / val_value
    // -----------------------------------------------------------------------

    #[test]
    fn monitored_metric_train_value_returns_correct_field() {
        let logs = CallbackLogs {
            loss: Some(0.5),
            accuracy: Some(0.8),
            precision: Some(0.7),
            recall: Some(0.6),
            f1: Some(0.65),
            ..CallbackLogs::default()
        };
        assert_eq!(MonitoredMetric::Loss.train_value(&logs), Some(0.5));
        assert_eq!(MonitoredMetric::Accuracy.train_value(&logs), Some(0.8));
        assert_eq!(MonitoredMetric::Precision.train_value(&logs), Some(0.7));
        assert_eq!(MonitoredMetric::Recall.train_value(&logs), Some(0.6));
        assert_eq!(MonitoredMetric::F1.train_value(&logs), Some(0.65));
    }

    #[test]
    fn monitored_metric_val_value_returns_correct_field() {
        let logs = CallbackLogs {
            val_loss: Some(0.4),
            val_accuracy: Some(0.9),
            val_precision: Some(0.85),
            val_recall: Some(0.75),
            val_f1: Some(0.80),
            ..CallbackLogs::default()
        };
        assert_eq!(MonitoredMetric::Loss.val_value(&logs), Some(0.4));
        assert_eq!(MonitoredMetric::Accuracy.val_value(&logs), Some(0.9));
        assert_eq!(MonitoredMetric::Precision.val_value(&logs), Some(0.85));
        assert_eq!(MonitoredMetric::Recall.val_value(&logs), Some(0.75));
        assert_eq!(MonitoredMetric::F1.val_value(&logs), Some(0.80));
    }

    // -----------------------------------------------------------------------
    // parse_monitored_metrics
    // -----------------------------------------------------------------------

    #[test]
    fn parse_monitored_metrics_parses_csv_list() {
        let result = parse_monitored_metrics("loss,accuracy").unwrap();
        assert_eq!(
            result,
            vec![MonitoredMetric::Loss, MonitoredMetric::Accuracy]
        );
    }

    #[test]
    fn parse_monitored_metrics_deduplicates() {
        let result = parse_monitored_metrics("loss,loss").unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn parse_monitored_metrics_rejects_unknown() {
        assert!(parse_monitored_metrics("loss,bogus").is_err());
    }

    #[test]
    fn parse_monitored_metrics_rejects_empty_input() {
        // A single unknown token triggers error
        assert!(parse_monitored_metrics("").is_err());
    }

    // -----------------------------------------------------------------------
    // MonitorMode::parse
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // HistoryCallback
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // EarlyStoppingCallback
    // -----------------------------------------------------------------------

    fn make_cfg(mode: MonitorMode, patience: usize) -> EarlyStoppingConfig {
        EarlyStoppingConfig {
            enabled: true,
            metric: MonitoredMetric::Loss,
            mode,
            patience,
            min_delta: 0.0,
            start_epoch: 0,
        }
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
    fn early_stopping_stops_after_patience_exceeded() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 2));
        // epoch 0: best = 1.0
        cb.on_epoch_end(0, Some(&loss_logs(1.0)));
        // epochs 1..=3: no improvement (wait goes 1, 2, 3 > patience=2 → stop)
        cb.on_epoch_end(1, Some(&loss_logs(1.0)));
        cb.on_epoch_end(2, Some(&loss_logs(1.0)));
        cb.on_epoch_end(3, Some(&loss_logs(1.0)));
        assert!(cb.stopped());
    }

    #[test]
    fn early_stopping_respects_start_epoch() {
        let mut cfg = make_cfg(MonitorMode::Min, 1);
        cfg.start_epoch = 5;
        let mut cb = EarlyStoppingCallback::new(cfg);
        // epochs 0..=4: below start_epoch, should be ignored
        for epoch in 0..5 {
            cb.on_epoch_end(epoch, Some(&loss_logs(2.0)));
        }
        assert!(!cb.stopped());
    }

    #[test]
    fn early_stopping_respects_max_mode() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Max, 1));
        // accuracy going up: should not stop
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
    fn early_stopping_best_epoch_tracks_best() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 5));
        cb.on_epoch_end(0, Some(&loss_logs(1.0)));
        cb.on_epoch_end(1, Some(&loss_logs(0.5)));
        cb.on_epoch_end(2, Some(&loss_logs(0.8)));
        assert_eq!(cb.best_epoch(), Some(1));
    }

    #[test]
    fn early_stopping_ignores_epoch_end_with_no_logs() {
        let mut cb = EarlyStoppingCallback::new(make_cfg(MonitorMode::Min, 1));
        cb.on_epoch_end(0, None);
        assert!(!cb.stopped());
    }

    #[test]
    fn early_stopping_returns_early_when_monitored_metric_absent_from_logs() {
        // Monitor Accuracy but provide only loss in logs → current = None → early return.
        let mut cfg = make_cfg(MonitorMode::Min, 1);
        cfg.metric = MonitoredMetric::Accuracy;
        let mut cb = EarlyStoppingCallback::new(cfg);
        let logs = CallbackLogs {
            loss: Some(0.5),
            ..CallbackLogs::default()
        };
        cb.on_epoch_end(0, Some(&logs));
        // The callback should not have started tracking (returned early before best_value set).
        assert!(!cb.stopped());
        assert!(cb.best_epoch().is_none());
    }
}
