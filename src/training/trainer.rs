use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::Rng;
use rand::seq::SliceRandom;

use crate::{
    console::{Tone, bold, paint},
    network::{
        callbacks::{Callback, CallbackLogs},
        model::Network,
    },
    training::{
        batch::create_batches,
        loss::{LossFunction, assert_binary_output},
        metrics::{Metrics, compute_classification_scores_from_labels},
        optimizer::Optimizer,
    },
};

fn make_epoch_logs(metrics: &Metrics, has_val: bool) -> CallbackLogs {
    CallbackLogs {
        loss: Some(metrics.train_loss),
        val_loss: has_val.then_some(metrics.val_loss),
        accuracy: Some(metrics.train_accuracy),
        val_accuracy: has_val.then_some(metrics.val_accuracy),
        precision: Some(metrics.train_precision),
        val_precision: has_val.then_some(metrics.val_precision),
    }
}

fn mean_epoch_loss(total_loss: f64, batch_count: usize) -> f64 {
    if batch_count == 0 {
        0.0
    } else {
        total_loss / (batch_count as f64)
    }
}

fn begin_epoch(callbacks: &mut [&mut dyn Callback], epoch: usize) {
    for callback in callbacks.iter_mut() {
        callback.on_epoch_begin(epoch);
    }
}

fn end_epoch(
    callbacks: &mut [&mut dyn Callback],
    epoch: usize,
    logs: &CallbackLogs,
    network: &mut Network,
) {
    for callback in callbacks.iter_mut() {
        callback.on_epoch_end(epoch, Some(logs));
    }
    for callback in callbacks.iter_mut() {
        callback.on_epoch_end_network(epoch, Some(logs), network);
    }
}

fn print_early_stop(epoch: usize) {
    println!(
        "{} {}",
        paint("Early stopping:", Tone::Warn),
        paint(&format!("stopped at epoch {}", epoch + 1), Tone::Warn)
    );
}

fn print_summary(metrics: &Metrics, has_val: bool) {
    println!(
        "{} {} - {}",
        bold(&paint("Training finished", Tone::Success)),
        paint(
            &format!("accuracy={:.4}", metrics.train_accuracy),
            Tone::TrainMetric
        ),
        paint(
            &format!("precision={:.4}", metrics.train_precision),
            Tone::TrainMetric
        ),
    );
    if has_val {
        println!(
            "{} {} - {}",
            paint("Validation:", Tone::Info),
            paint(
                &format!("val_accuracy={:.4}", metrics.val_accuracy),
                Tone::ValMetric
            ),
            paint(
                &format!("val_precision={:.4}", metrics.val_precision),
                Tone::ValMetric
            ),
        );
    }
}

pub struct Trainer {
    optimizer: Box<dyn Optimizer>,
    loss_fn: LossFunction,
    batch_size: usize,
    epochs: usize,
}

struct EpochContext<'n, 'd, 'i, 'm, 'c, R: Rng> {
    network: &'n mut Network,
    x_train: ArrayView2<'d, f64>,
    y_train: ArrayView1<'d, f64>,
    val_data: Option<(ArrayView2<'d, f64>, ArrayView1<'d, f64>)>,
    indices: &'i mut [usize],
    rng: &'i mut R,
    metrics: &'m mut Metrics,
    callbacks: &'m mut [&'c mut dyn Callback],
    epoch: usize,
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

    /// train: full loop over epochs/batches, each batch runs apply_batch.
    pub fn train<'data>(
        &mut self,
        network: &mut Network,
        x_train: ArrayView2<'data, f64>,
        y_train: ArrayView1<'data, f64>,
        val_data: Option<(ArrayView2<'data, f64>, ArrayView1<'data, f64>)>,
        callbacks: &mut [&mut dyn Callback],
    ) -> Metrics {
        assert_binary_output(network.output_size());
        let mut metrics = Metrics::default();
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..x_train.nrows()).collect();

        for callback in callbacks.iter_mut() {
            callback.on_train_begin();
        }

        for epoch in 0..self.epochs {
            if self.run_training_epoch(EpochContext {
                network,
                x_train,
                y_train,
                val_data,
                indices: &mut indices,
                rng: &mut rng,
                metrics: &mut metrics,
                callbacks,
                epoch,
            }) {
                break;
            }
        }

        for callback in callbacks.iter_mut() {
            callback.on_train_end();
        }

        print_summary(&metrics, val_data.is_some());

        metrics
    }

    fn run_training_epoch<'d, R: Rng>(&mut self, ctx: EpochContext<'_, 'd, '_, '_, '_, R>) -> bool {
        let EpochContext {
            network,
            x_train,
            y_train,
            val_data,
            indices,
            rng,
            metrics,
            callbacks,
            epoch,
        } = ctx;
        begin_epoch(callbacks, epoch);
        indices.shuffle(rng);

        let (total_loss, batch_index) = self.run_epoch(network, x_train, y_train, indices);
        metrics.train_loss = mean_epoch_loss(total_loss, batch_index);
        self.eval_epoch(network, x_train, y_train, val_data, metrics);

        let epoch_logs = make_epoch_logs(metrics, val_data.is_some());
        end_epoch(callbacks, epoch, &epoch_logs, network);

        if callbacks.iter().any(|callback| callback.should_stop()) {
            print_early_stop(epoch);
            return true;
        }
        false
    }

    fn run_epoch<'data>(
        &mut self,
        network: &mut Network,
        x_train: ArrayView2<'data, f64>,
        y_train: ArrayView1<'data, f64>,
        indices: &[usize],
    ) -> (f64, usize) {
        let mut batch_index = 0usize;
        let mut total_loss = 0.0;

        create_batches(x_train, y_train, indices, self.batch_size).sum_by(|x_batch, y_batch| {
            let batch_loss = self.apply_batch(network, &x_batch, &y_batch);
            total_loss += batch_loss;
            batch_index += 1;
            batch_loss
        });

        (total_loss, batch_index)
    }

    /// apply_batch: forward, compute loss, backward (softmax → p − y, else BCE gradient gated), apply optimizer update.
    fn apply_batch(
        &mut self,
        network: &mut Network,
        x: &Array2<f64>,
        t: &Array1<f64>,
    ) -> f64 {
        let a = network.forward(x);
        let loss = self
            .loss_fn
            .compute(a.view(), t.view())
            .mean()
            .unwrap_or(0.0);
        let grads = network.backward(self.loss_fn, t.view());
        self.optimizer.update(network, &grads);
        loss
    }

    fn evaluate<'data>(
        &self,
        network: &mut Network,
        x: ArrayView2<'data, f64>,
        y: ArrayView1<'data, f64>,
    ) -> (f64, f64, f64) {
        let pred = network.forward(x);
        let loss = self.loss_fn.compute(pred.view(), y).mean().unwrap_or(0.0);
        let scores = compute_classification_scores_from_labels(pred.view(), y);
        (loss, scores.accuracy, scores.precision)
    }

    fn eval_epoch<'data>(
        &self,
        network: &mut Network,
        x_train: ArrayView2<'data, f64>,
        y_train: ArrayView1<'data, f64>,
        val_data: Option<(ArrayView2<'data, f64>, ArrayView1<'data, f64>)>,
        metrics: &mut Metrics,
    ) {
        let (_, accuracy, precision) = self.evaluate(network, x_train, y_train);
        metrics.train_accuracy = accuracy;
        metrics.train_precision = precision;

        if let Some((x_val, y_val)) = val_data {
            let (loss, accuracy, precision) = self.evaluate(network, x_val, y_val);
            metrics.val_loss = loss;
            metrics.val_accuracy = accuracy;
            metrics.val_precision = precision;
        } else {
            metrics.val_loss = 0.0;
            metrics.val_accuracy = 0.0;
            metrics.val_precision = 0.0;
        }
    }
}
