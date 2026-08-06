use mlp::training::monitor::MonitoredMetric;
use mlp::visualization::plotter::{TrainingHistory, plot_training_curves};

fn history_with_metrics() -> TrainingHistory {
    TrainingHistory {
        train_loss: vec![0.9, 0.5, 0.2],
        val_loss: vec![1.0, 0.6, 0.3],
        train_accuracy: vec![0.5, 0.8, 0.95],
        val_accuracy: vec![0.45, 0.75, 0.9],
        train_precision: vec![0.4, 0.7, 0.9],
        val_precision: vec![0.35, 0.65, 0.85],
    }
}

fn empty_history() -> TrainingHistory {
    TrainingHistory {
        train_loss: vec![],
        val_loss: vec![],
        train_accuracy: vec![],
        val_accuracy: vec![],
        train_precision: vec![],
        val_precision: vec![],
    }
}

#[test]
fn plot_training_curves_returns_error_when_loss_is_empty() {
    let mut history = empty_history();
    history.train_accuracy = vec![0.5];
    let result = plot_training_curves(&history, "/tmp/mlp_test_curves_empty.png", &[]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));
}

#[test]
fn plot_training_curves_returns_error_when_loss_is_non_finite() {
    let mut history = empty_history();
    history.train_loss = vec![f64::NAN];
    history.train_accuracy = vec![0.5];
    let result = plot_training_curves(&history, "/tmp/mlp_test_curves_inf.png", &[]);
    assert!(result.is_err());
}

#[test]
fn plot_training_curves_saves_file() {
    let history = history_with_metrics();
    let path = format!("/tmp/mlp_test_plot_ok_curves_{}.png", std::process::id());
    plot_training_curves(&history, &path, &[]).expect("combined curves should save");
    assert!(std::path::Path::new(&path).exists());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn plot_training_curves_honors_metric_selection() {
    let history = history_with_metrics();
    let path = format!(
        "/tmp/mlp_test_plot_ok_curves_prec_{}.png",
        std::process::id()
    );
    plot_training_curves(&history, &path, &[MonitoredMetric::Precision])
        .expect("precision-only curves should save");
    assert!(std::path::Path::new(&path).exists());
    let _ = std::fs::remove_file(&path);
}
