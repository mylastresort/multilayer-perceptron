use std::collections::BTreeSet;

use ndarray::{Array2, ArrayView1, ArrayView2};

#[derive(Debug, Default)]
pub struct Metrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub train_precision: f64,
    pub val_precision: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClassificationScores {
    pub accuracy: f64,
    pub precision: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClassMetrics {
    pub class_id: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

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

fn class_outcomes(
    pred_labels: &[usize],
    true_labels: &[usize],
    class_id: usize,
) -> (usize, usize, usize) {
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
    (tp, fp, fn_)
}

fn ratio_or_zero(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / (denominator as f64)
    }
}

fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn class_metrics(pred_labels: &[usize], true_labels: &[usize], class_id: usize) -> ClassMetrics {
    let (tp, fp, fn_) = class_outcomes(pred_labels, true_labels, class_id);
    let precision = ratio_or_zero(tp, tp + fp);
    let recall = ratio_or_zero(tp, tp + fn_);
    ClassMetrics {
        class_id,
        precision,
        recall,
        f1: f1_score(precision, recall),
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
        ClassMetrics, accuracy, compute_classification_report,
        compute_classification_scores_from_labels,
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
        let targets = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let acc = accuracy(&preds, &targets);
        assert!((acc - 1.0).abs() < 1e-9);
    }
}
