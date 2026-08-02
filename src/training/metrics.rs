use std::collections::BTreeSet;

use ndarray::{Array2, ArrayView1, ArrayView2};

/// Aggregated metrics from a training epoch (train and validation).
#[derive(Debug, Default)]
pub struct Metrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub train_precision: f64,
    pub val_precision: f64,
}

/// Per-class classification scores (macro-averaged across classes).
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassificationScores {
    pub accuracy: f64,
    pub precision: f64,
}

/// Precision, recall and F1 for a single class.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassMetrics {
    pub class_id: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Per-class classification report plus overall accuracy.
#[derive(Debug, Clone, Default)]
pub struct ClassificationReport {
    pub classes: Vec<ClassMetrics>,
    pub accuracy: f64,
}

fn predicted_class_from_row(row: ndarray::ArrayView1<'_, f64>) -> usize {
    if row.len() == 1 {
        if row[0] >= 0.5 { 1 } else { 0 }
    } else {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

fn class_metrics(pred_labels: &[usize], true_labels: &[usize], class_id: usize) -> ClassMetrics {
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    for (pred, true_) in pred_labels.iter().zip(true_labels.iter()) {
        if *pred == class_id && *true_ == class_id {
            tp += 1;
        } else if *pred == class_id {
            fp += 1;
        } else if *true_ == class_id {
            fn_ += 1;
        }
    }
    let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / ((tp + fp) as f64) };
    let recall = if tp + fn_ == 0 { 0.0 } else { tp as f64 / ((tp + fn_) as f64) };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    ClassMetrics {
        class_id,
        precision,
        recall,
        f1,
        support: tp + fn_,
    }
}

fn class_precision(pred_labels: &[usize], true_labels: &[usize], class_id: usize) -> f64 {
    class_metrics(pred_labels, true_labels, class_id).precision
}

fn macro_precision(pred_labels: &[usize], true_labels: &[usize]) -> f64 {
    let classes: BTreeSet<usize> = pred_labels
        .iter()
        .copied()
        .chain(true_labels.iter().copied())
        .collect();
    if classes.is_empty() {
        return 0.0;
    }
    classes
        .iter()
        .copied()
        .map(|class_id| class_precision(pred_labels, true_labels, class_id))
        .sum::<f64>()
        / (classes.len() as f64)
}

/// Computes macro-averaged accuracy and precision from predictions and labels.
pub fn compute_classification_scores_from_labels(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> ClassificationScores {
    if predictions.nrows() == 0 || predictions.nrows() != targets.len() {
        return ClassificationScores::default();
    }

    let pred_labels: Vec<usize> = predictions
        .outer_iter()
        .map(predicted_class_from_row)
        .collect();
    let true_labels: Vec<usize> = targets.iter().map(|t| t.round() as usize).collect();

    let correct = pred_labels
        .iter()
        .zip(true_labels.iter())
        .filter(|(p, t)| *p == *t)
        .count();

    ClassificationScores {
        accuracy: correct as f64 / (predictions.nrows() as f64),
        precision: macro_precision(&pred_labels, &true_labels),
    }
}

/// Computes a per-class classification report (precision/recall/F1/support).
pub fn compute_classification_report(
    predictions: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
) -> ClassificationReport {
    if predictions.nrows() == 0 || predictions.nrows() != targets.len() {
        return ClassificationReport::default();
    }

    let pred_labels: Vec<usize> = predictions
        .outer_iter()
        .map(predicted_class_from_row)
        .collect();
    let true_labels: Vec<usize> = targets.iter().map(|t| t.round() as usize).collect();

    let classes: BTreeSet<usize> = pred_labels
        .iter()
        .copied()
        .chain(true_labels.iter().copied())
        .collect();
    let correct = pred_labels
        .iter()
        .zip(true_labels.iter())
        .filter(|(p, t)| *p == *t)
        .count();

    ClassificationReport {
        classes: classes
            .iter()
            .copied()
            .map(|class_id| class_metrics(&pred_labels, &true_labels, class_id))
            .collect(),
        accuracy: correct as f64 / (predictions.nrows() as f64),
    }
}

/// Computes accuracy from 2D prediction arrays and 2D target arrays.
pub fn accuracy(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    if predictions.nrows() == 0 || predictions.nrows() != targets.nrows() {
        return 0.0;
    }

    let target_labels: Vec<f64> = if targets.ncols() == 1 {
        targets.column(0).iter().copied().collect()
    } else {
        targets
            .outer_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(idx, _)| idx as f64)
                    .unwrap_or(0.0)
            })
            .collect()
    };

    let target_labels_arr = ndarray::Array1::from(target_labels);
    compute_classification_scores_from_labels(predictions.view(), target_labels_arr.view()).accuracy
}

#[cfg(test)]
mod tests {
    use super::{
        accuracy, compute_classification_report, compute_classification_scores_from_labels,
        ClassMetrics,
    };
    use ndarray::{arr1, arr2};

    // -----------------------------------------------------------------------
    // compute_classification_scores_from_labels
    // -----------------------------------------------------------------------

    #[test]
    fn scores_empty_predictions_returns_default() {
        let preds = arr2(&[[0.0_f64; 2]; 0]); // 0 rows
        let targets = arr1(&[0.0_f64; 0]);
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert_eq!(scores.accuracy, 0.0);
        assert_eq!(scores.precision, 0.0);
    }

    #[test]
    fn scores_mismatched_rows_returns_default() {
        let preds = arr2(&[[0.8, 0.2], [0.3, 0.7]]);
        let targets = arr1(&[0.0]); // only 1 target for 2 rows
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert_eq!(scores.accuracy, 0.0);
    }

    #[test]
    fn scores_binary_single_column_perfect_classification() {
        // Single-output: ≥ 0.5 → class 1, < 0.5 → class 0
        let preds = arr2(&[[0.9], [0.1], [0.8], [0.2]]);
        let targets = arr1(&[1.0, 0.0, 1.0, 0.0]);
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert!((scores.accuracy - 1.0).abs() < 1e-9);
        assert!((scores.precision - 1.0).abs() < 1e-9);
    }

    #[test]
    fn scores_binary_single_column_partial_errors() {
        // pred 0.6 vs target 0.0 is a false positive
        let preds = arr2(&[[0.9], [0.6], [0.1], [0.2]]);
        let targets = arr1(&[1.0, 0.0, 0.0, 0.0]);
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert!(scores.accuracy < 1.0);
    }

    #[test]
    fn scores_multiclass_perfect_classification() {
        // 3 classes: argmax picks correct class each time
        let preds = arr2(&[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]);
        let targets = arr1(&[0.0, 1.0, 2.0]);
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert!((scores.accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn scores_all_wrong_predictions_zero_accuracy() {
        let preds = arr2(&[[0.1, 0.9], [0.1, 0.9]]);
        let targets = arr1(&[0.0, 0.0]); // correct class is 0, pred always 1
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert_eq!(scores.accuracy, 0.0);
    }

    #[test]
    fn scores_precision_zero_when_no_tp_and_fp_match() {
        // No predicted positive for class 1, tp+fp = 0 → precision = 0
        let preds = arr2(&[[0.9, 0.1], [0.9, 0.1]]);
        let targets = arr1(&[1.0, 1.0]);
        let scores = compute_classification_scores_from_labels(preds.view(), targets.view());
        assert_eq!(scores.accuracy, 0.0);
    }

    // -----------------------------------------------------------------------
    // compute_classification_report
    // -----------------------------------------------------------------------

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
        // pred: 0,1,1,0   true: 1,1,0,0
        // class 1: tp=1 (row1), fp=1 (row2), fn=1 (row0) → p=0.5, r=0.5, f1=0.5, support=2
        // class 0: tp=1 (row3), fp=1 (row0), fn=1 (row2) → p=0.5, r=0.5, f1=0.5, support=2
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
        // pred: 0,1,2,0  true: 0,2,2,1  (2 correct → accuracy 0.5)
        // class 0: tp=1 (row0), fp=1 (row3), fn=0 → p=0.5, r=1.0, support=1
        // class 1: tp=0, fp=1 (row1), fn=1 (row3) → p=0, r=0, support=1
        // class 2: tp=1 (row2), fp=0, fn=1 (row1) → p=1.0, r=0.5, support=2
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

    // -----------------------------------------------------------------------
    // accuracy function
    // -----------------------------------------------------------------------

    #[test]
    fn accuracy_empty_predictions_returns_zero() {
        let preds = arr2(&[[0.0_f64; 2]; 0]);
        let targets = arr2(&[[0.0_f64; 2]; 0]);
        assert_eq!(accuracy(&preds, &targets), 0.0);
    }

    #[test]
    fn accuracy_mismatched_rows_returns_zero() {
        let preds = arr2(&[[0.8, 0.2]]);
        let targets = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(accuracy(&preds, &targets), 0.0);
    }

    #[test]
    fn accuracy_single_column_target() {
        let preds = arr2(&[[0.9], [0.1]]);
        let targets = arr2(&[[1.0], [0.0]]);
        let acc = accuracy(&preds, &targets);
        assert!((acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn accuracy_multiclass_target_one_hot_style() {
        let preds = arr2(&[[0.8, 0.2], [0.3, 0.7]]);
        // targets as one-hot encoded (argmax picks col 0 and col 1 respectively)
        let targets = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let acc = accuracy(&preds, &targets);
        assert!((acc - 1.0).abs() < 1e-9);
    }
}
