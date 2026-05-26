use ndarray::{ArrayView1, ArrayView2};
use rand::seq::SliceRandom;

use crate::{
    console::{Tone, bold, paint},
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
                    precision: None,
                    val_precision: None,
                    recall: None,
                    val_recall: None,
                    f1: None,
                    val_f1: None,
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
                precision: Some(metrics.train_precision),
                val_precision: val_data.map(|_| metrics.val_precision),
                recall: Some(metrics.train_recall),
                val_recall: val_data.map(|_| metrics.val_recall),
                f1: Some(metrics.train_f1),
                val_f1: val_data.map(|_| metrics.val_f1),
            };
            for callback in callbacks.iter_mut() {
                callback.on_epoch_end(epoch, Some(&epoch_logs));
            }

            if callbacks.iter().any(|callback| callback.should_stop()) {
                println!(
                    "{} {}",
                    paint("Early stopping:", Tone::Warn),
                    paint(&format!("stopped at epoch {}", epoch + 1), Tone::Warn)
                );
                break;
            }
        }

        for callback in callbacks.iter_mut() {
            callback.on_train_end();
        }

        println!(
            "{} {} - {} - {} - {}",
            bold(&paint("Training finished", Tone::Success)),
            paint(
                &format!("accuracy={:.4}", metrics.train_accuracy),
                Tone::TrainMetric
            ),
            paint(
                &format!("precision={:.4}", metrics.train_precision),
                Tone::TrainMetric
            ),
            paint(
                &format!("recall={:.4}", metrics.train_recall),
                Tone::TrainMetric
            ),
            paint(&format!("f1={:.4}", metrics.train_f1), Tone::TrainMetric)
        );
        if val_data.is_some() {
            println!(
                "{} {} - {} - {} - {}",
                paint("Validation:", Tone::Info),
                paint(
                    &format!("val_accuracy={:.4}", metrics.val_accuracy),
                    Tone::ValMetric
                ),
                paint(
                    &format!("val_precision={:.4}", metrics.val_precision),
                    Tone::ValMetric
                ),
                paint(
                    &format!("val_recall={:.4}", metrics.val_recall),
                    Tone::ValMetric
                ),
                paint(&format!("val_f1={:.4}", metrics.val_f1), Tone::ValMetric)
            );
        }

        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::Trainer;
    use crate::network::{
        activation::ActivationFunction,
        callbacks::{Callback, CallbackLogs},
        initializer::WeightInitializer,
        layer::Layer,
        model::Network,
    };
    use crate::training::{
        loss::LossFunction,
        optimizer::{Optimizer, OptimizerType, SGD},
    };
    use ndarray::{Array1, Array2};

    fn tiny_net() -> Network {
        Network::new()
            .add_layer(Layer::new(
                2,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .build()
    }

    fn tiny_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64 * 0.1);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        (x, y)
    }

    /// Sets `seen_epoch_end = true` in `on_epoch_end`; `should_stop()` returns that flag.
    /// This ensures the early-stopping branch fires on the very first epoch.
    struct StopAfterFirstEpoch {
        seen_epoch_end: bool,
    }
    impl Callback for StopAfterFirstEpoch {
        fn on_epoch_end(&mut self, _epoch: usize, _logs: Option<&CallbackLogs>) {
            self.seen_epoch_end = true;
        }
        fn should_stop(&self) -> bool {
            self.seen_epoch_end
        }
    }

    #[test]
    fn trainer_default_has_sensible_fields() {
        let t = Trainer::default();
        drop(t);
    }

    #[test]
    fn trainer_from_optimizer_builds() {
        let t = Trainer::from(SGD::new(0.01));
        drop(t);
    }

    #[test]
    fn trainer_from_boxed_optimizer_builds() {
        let boxed: Box<dyn Optimizer> = Box::new(SGD::new(0.01));
        let t = Trainer::from(boxed);
        drop(t);
    }

    #[test]
    fn trainer_set_learning_rate_does_not_panic() {
        let mut t = Trainer::from(SGD::new(0.01));
        t.set_learning_rate(0.001);
    }

    #[test]
    fn trainer_train_one_epoch_returns_metrics() {
        let mut net = tiny_net();
        let (x, y) = tiny_data();
        let mut t = Trainer::new(
            OptimizerType::SGD.create(0.01),
            LossFunction::BinaryCrossEntropy,
            4,
            1,
        );
        let metrics = t.train(&mut net, x.view(), y.view(), None, &mut []);
        assert!(metrics.train_loss.is_finite());
        assert!(metrics.train_accuracy >= 0.0 && metrics.train_accuracy <= 1.0);
    }

    #[test]
    fn trainer_train_with_validation_data_sets_val_metrics() {
        let mut net = tiny_net();
        let (x, y) = tiny_data();
        let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::MSE, 4, 1);
        let val = (x.view(), y.view());
        let metrics = t.train(&mut net, x.view(), y.view(), Some(val), &mut []);
        assert!(metrics.val_loss.is_finite());
    }

    #[test]
    fn trainer_train_without_validation_data_zeroes_val_metrics() {
        let mut net = tiny_net();
        let (x, y) = tiny_data();
        let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::MSE, 4, 1);
        let metrics = t.train(&mut net, x.view(), y.view(), None, &mut []);
        assert_eq!(metrics.val_loss, 0.0);
        assert_eq!(metrics.val_accuracy, 0.0);
    }

    #[test]
    fn trainer_early_stopping_branch_fires_and_on_train_end_called() {
        // StopAfterFirstEpoch: on_epoch_end sets seen=true → should_stop() returns true
        // → the early-stopping println+break block executes, then on_train_end runs.
        let mut net = tiny_net();
        let (x, y) = tiny_data();
        let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::MSE, 4, 5);
        let mut cb = StopAfterFirstEpoch {
            seen_epoch_end: false,
        };
        let metrics = t.train(
            &mut net,
            x.view(),
            y.view(),
            None,
            &mut [&mut cb as &mut dyn Callback],
        );
        assert!(metrics.train_loss.is_finite());
    }

    #[test]
    fn trainer_train_with_empty_data_returns_zero_loss() {
        let mut net = tiny_net();
        let x = Array2::<f64>::zeros((0, 2));
        let y = Array1::<f64>::zeros(0);
        let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::MSE, 4, 1);
        let metrics = t.train(&mut net, x.view(), y.view(), None, &mut []);
        assert_eq!(metrics.train_loss, 0.0);
    }
}
