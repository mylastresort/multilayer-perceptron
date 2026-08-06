use mlp::training::metrics::{
    ClassMetrics, compute_classification_report, compute_classification_scores_from_labels,
};
use ndarray::{arr1, arr2};

#[test]
fn scores_empty_predictions_returns_default() {
    let preds = arr2(&[[0.0_f64; 2]; 0]);
    let targets = arr1(&[0.0_f64; 0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert_eq!(scores.accuracy, 0.0);
    assert_eq!(scores.precision, 0.0);
}

#[test]
fn scores_mismatched_rows_returns_default() {
    let preds = arr2(&[[0.8, 0.2], [0.3, 0.7]]);
    let targets = arr1(&[0.0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert_eq!(scores.accuracy, 0.0);
}

#[test]
fn scores_binary_single_column_perfect_classification() {
    let preds = arr2(&[[0.9], [0.1], [0.8], [0.2]]);
    let targets = arr1(&[1.0, 0.0, 1.0, 0.0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert!((scores.accuracy - 1.0).abs() < 1e-9);
    assert!((scores.precision - 1.0).abs() < 1e-9);
}

#[test]
fn scores_binary_single_column_partial_errors() {
    let preds = arr2(&[[0.9], [0.6], [0.1], [0.2]]);
    let targets = arr1(&[1.0, 0.0, 0.0, 0.0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert!(scores.accuracy < 1.0);
}

#[test]
fn scores_multiclass_perfect_classification() {
    let preds = arr2(&[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]);
    let targets = arr1(&[0.0, 1.0, 2.0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert!((scores.accuracy - 1.0).abs() < 1e-9);
}

#[test]
fn scores_all_wrong_predictions_zero_accuracy() {
    let preds = arr2(&[[0.1, 0.9], [0.1, 0.9]]);
    let targets = arr1(&[0.0, 0.0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert_eq!(scores.accuracy, 0.0);
}

#[test]
fn scores_precision_zero_when_no_tp_and_fp_match() {
    let preds = arr2(&[[0.9, 0.1], [0.9, 0.1]]);
    let targets = arr1(&[1.0, 1.0]);
    let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
    assert_eq!(scores.accuracy, 0.0);
}

#[test]
fn report_empty_predictions_returns_default() {
    let preds = arr2(&[[0.0_f64; 2]; 0]);
    let targets = arr1(&[0.0_f64; 0]);
    let report = compute_classification_report(preds.view(), targets.view());
    assert!(report.classes.is_empty());
    assert_eq!(report.accuracy, 0.0);
}

#[test]
fn report_binary_perfect_classification() {
    let preds = arr2(&[[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]);
    let targets = arr1(&[0.0, 1.0, 0.0]);
    let report = compute_classification_report(preds.view(), targets.view());
    assert!((report.accuracy - 1.0).abs() < 1e-9);
    assert_eq!(report.classes.len(), 2);
    for c in &report.classes {
        assert!((c.precision - 1.0).abs() < 1e-9);
        assert!((c.recall - 1.0).abs() < 1e-9);
        assert!((c.f1 - 1.0).abs() < 1e-9);
    }
}

#[test]
fn report_binary_known_errors() {
    let preds = arr2(&[[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]]);
    let targets = arr1(&[1.0, 1.0, 0.0, 0.0]);
    let report = compute_classification_report(preds.view(), targets.view());
    assert!((report.accuracy - 0.5).abs() < 1e-9);
    assert_eq!(report.classes.len(), 2);
    for c in &report.classes {
        assert!((c.precision - 0.5).abs() < 1e-9);
        assert!((c.recall - 0.5).abs() < 1e-9);
        assert!((c.f1 - 0.5).abs() < 1e-9);
        assert_eq!(c.support, 2);
    }
}

#[test]
fn report_multiclass_metrics() {
    let preds = arr2(&[
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
    ]);
    let targets = arr1(&[0.0, 2.0, 2.0, 1.0]);
    let report = compute_classification_report(preds.view(), targets.view());
    assert_eq!(report.classes.len(), 3);
    assert!((report.accuracy - 0.5).abs() < 1e-9);

    let by_class = |id: usize| -> &ClassMetrics {
        report.classes.iter().find(|c| c.class_id == id).unwrap()
    };
    let c0 = by_class(0);
    assert!((c0.precision - 0.5).abs() < 1e-9);
    assert!((c0.recall - 1.0).abs() < 1e-9);
    assert_eq!(c0.support, 1);
    let c1 = by_class(1);
    assert_eq!(c1.precision, 0.0);
    assert_eq!(c1.recall, 0.0);
    assert_eq!(c1.f1, 0.0);
    assert_eq!(c1.support, 1);
    let c2 = by_class(2);
    assert!((c2.precision - 1.0).abs() < 1e-9);
    assert!((c2.recall - 0.5).abs() < 1e-9);
    assert_eq!(c2.support, 2);
}
