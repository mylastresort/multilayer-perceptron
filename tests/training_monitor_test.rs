use mlp::network::callbacks::{Callback, CallbackLogs};
use mlp::network::model::Network;
use mlp::training::monitor::{
    EarlyStoppingCallback, EarlyStoppingConfig, HistoryCallback, MonitoredMetric, MonitorMode,
};
use ndarray::Array2;

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
        metric: MonitoredMetric::Loss,
        mode,
        patience,
        min_delta: 0.0,
        start_epoch: 0,
        restore_best_weights: true,
    }
}

fn test_network() -> Network {
    use mlp::network::{
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
