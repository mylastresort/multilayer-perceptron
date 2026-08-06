use std::collections::BTreeSet;

use ndarray::{ArrayView1, ArrayView2};

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
