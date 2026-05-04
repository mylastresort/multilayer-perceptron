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
    pub train_recall: f64,
    pub val_recall: f64,
    pub train_f1: f64,
    pub val_f1: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClassificationScores {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
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
    let accuracy = correct as f64 / (predictions.nrows() as f64);

    let classes: BTreeSet<usize> = pred_labels
        .iter()
        .copied()
        .chain(true_labels.iter().copied())
        .collect();
    if classes.is_empty() {
        return ClassificationScores {
            accuracy,
            ..ClassificationScores::default()
        };
    }

    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut f1_sum = 0.0;

    for class_id in classes.iter().copied() {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (pred, true_) in pred_labels.iter().zip(true_labels.iter()) {
            match (*pred == class_id, *true_ == class_id) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }

        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / ((tp + fp) as f64)
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / ((tp + fn_) as f64)
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }

    let class_count = classes.len() as f64;
    ClassificationScores {
        accuracy,
        precision: precision_sum / class_count,
        recall: recall_sum / class_count,
        f1_score: f1_sum / class_count,
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
