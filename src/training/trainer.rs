use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;

use crate::{
    network::{
        callbacks::{Callback, CallbackLogs},
        model::Network,
    },
    training::{
        batch::create_batches,
        loss::{Loss, LossFunction},
        metrics::{Metrics, compute_classification_scores_from_labels},
        optimizer::{Optimizer, SGD},
    },
};

pub struct Trainer {
    optimizer: Box<dyn Optimizer>,
    loss_fn: LossFunction,
    batch_size: usize,
    epochs: usize,
}

// #[derive(Serialize, Deserialize)]
// pub struct TrainingHistory {
//     pub train_loss: Vec<f64>,
//     pub val_loss: Vec<f64>,
//     pub train_accuracy: Vec<f64>,
//     pub val_accuracy: Vec<f64>,
// }

impl Default for Trainer {
    fn default() -> Self {
        Self {
            optimizer: Box::new(SGD::new(0.01)),
            loss_fn: LossFunction::MSE,
            batch_size: 32,
            epochs: 10,
        }
    }
}

// Allow constructing a Trainer from any boxed optimizer implementation.
impl From<Box<dyn Optimizer>> for Trainer {
    fn from(optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            optimizer,
            loss_fn: LossFunction::MSE,
            batch_size: 32,
            epochs: 10,
        }
    }
}

impl<T> From<T> for Trainer
where
    T: Optimizer + 'static,
{
    fn from(optimizer: T) -> Self {
        Self {
            optimizer: Box::new(optimizer),
            loss_fn: LossFunction::MSE,
            batch_size: 32,
            epochs: 10,
        }
    }
}

impl Trainer {
    pub fn new(
        optimizer: Box<dyn Optimizer>,
        loss_fn: LossFunction,
        batch_size: usize,
        epochs: usize,
    ) -> Self {
        Self {
            optimizer,
            loss_fn,
            batch_size,
            epochs,
        }
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    pub fn train<'data>(
        &mut self,
        network: &mut Network,
        x_train: ArrayView2<'data, f64>,
        y_train: ArrayView1<'data, f64>,
        val_data: Option<(ArrayView2<'data, f64>, ArrayView1<'data, f64>)>,
        callbacks: &mut [&mut dyn Callback],
    ) -> Metrics {
        let mut metrics = Metrics::default();
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..x_train.nrows()).collect();

        for callback in callbacks.iter_mut() {
            callback.on_train_begin();
        }

        for epoch in 0..self.epochs {
            for callback in callbacks.iter_mut() {
                callback.on_epoch_begin(epoch);
            }

            indices.shuffle(&mut rng);

            let mut batch_index = 0usize;
            let mut total_loss = 0.0;

            create_batches(x_train, y_train, &indices, self.batch_size).sum_by(|x_row, y_row| {
                for callback in callbacks.iter_mut() {
                    callback.on_batch_begin(batch_index);
                }

                let pred = network.forward(x_row);
                let batch_loss = self.loss_fn.compute(pred.view(), y_row);
                let loss_gradient = self.loss_fn.gradient(pred.view(), y_row);
                let gradients = network.backward(loss_gradient);
                self.optimizer.update(network, &gradients);

                total_loss += batch_loss;
                let batch_logs = CallbackLogs {
                    loss: Some(batch_loss),
                    val_loss: None,
                    accuracy: None,
                    val_accuracy: None,
                };
                for callback in callbacks.iter_mut() {
                    callback.on_batch_end(batch_index, Some(&batch_logs));
                }

                batch_index += 1;
                batch_loss
            });

            metrics.train_loss = if batch_index == 0 {
                0.0
            } else {
                total_loss / (batch_index as f64)
            };

            let train_pred = network.forward(x_train);
            let train_scores =
                compute_classification_scores_from_labels(train_pred.view(), y_train);
            metrics.train_accuracy = train_scores.accuracy;
            metrics.train_precision = train_scores.precision;
            metrics.train_recall = train_scores.recall;
            metrics.train_f1 = train_scores.f1_score;

            if let Some((x_val, y_val)) = val_data {
                let val_pred = network.forward(x_val);
                metrics.val_loss = self.loss_fn.compute(val_pred.view(), y_val);
                let val_scores = compute_classification_scores_from_labels(val_pred.view(), y_val);
                metrics.val_accuracy = val_scores.accuracy;
                metrics.val_precision = val_scores.precision;
                metrics.val_recall = val_scores.recall;
                metrics.val_f1 = val_scores.f1_score;
            } else {
                metrics.val_loss = 0.0;
                metrics.val_accuracy = 0.0;
                metrics.val_precision = 0.0;
                metrics.val_recall = 0.0;
                metrics.val_f1 = 0.0;
            }

            let epoch_logs = CallbackLogs {
                loss: Some(metrics.train_loss),
                val_loss: val_data.map(|_| metrics.val_loss),
                accuracy: Some(metrics.train_accuracy),
                val_accuracy: val_data.map(|_| metrics.val_accuracy),
            };
            for callback in callbacks.iter_mut() {
                callback.on_epoch_end(epoch, Some(&epoch_logs));
            }
        }

        for callback in callbacks.iter_mut() {
            callback.on_train_end();
        }

        println!(
            "Training finished - accuracy: {:.4} - precision: {:.4} - recall: {:.4} - f1_score: {:.4}",
            metrics.train_accuracy, metrics.train_precision, metrics.train_recall, metrics.train_f1
        );
        if val_data.is_some() {
            println!(
                "Validation metrics - val_accuracy: {:.4} - val_precision: {:.4} - val_recall: {:.4} - val_f1_score: {:.4}",
                metrics.val_accuracy, metrics.val_precision, metrics.val_recall, metrics.val_f1
            );
        }

        metrics
    }
}
