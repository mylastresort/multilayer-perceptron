use mlp::app::types::MonitorOptions;
use mlp::training::monitor::{MonitorMode, MonitoredMetric};

#[test]
fn monitor_options_default_has_sensible_values() {
    let opts = MonitorOptions::default();
    assert!(!opts.early_stopping);
    assert_eq!(opts.early_stop_patience, 60);
    assert!(matches!(opts.early_stop_mode, MonitorMode::Min));
    assert!(matches!(opts.early_stop_metric, MonitoredMetric::Loss));
    assert!(opts.history_out.is_none());
    assert_eq!(
        opts.metrics,
        vec![
            MonitoredMetric::Loss,
            MonitoredMetric::Accuracy,
            MonitoredMetric::Precision,
        ]
    );
}
