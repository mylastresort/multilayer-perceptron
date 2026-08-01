use crate::console::{Tone, bold, paint};
use crate::network::model::Network;

/// Aggregated metric values passed to callbacks at epoch/batch boundaries.
#[derive(Debug, Clone, Default)]
pub struct CallbackLogs {
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub val_precision: Option<f64>,
}

/// Prints epoch metrics to the console in a Keras-style format.
pub struct ProgressLogger {
    total_epochs: usize,
}

impl ProgressLogger {
    pub fn new(total_epochs: usize) -> Self {
        Self { total_epochs }
    }
}

impl Callback for ProgressLogger {
    fn on_epoch_end(&mut self, epoch: usize, logs: Option<&CallbackLogs>) {
        let Some(logs) = logs else {
            return;
        };

        let epoch_display = epoch + 1;

        let ordered_metrics: [(&str, Option<f64>); 6] = [
            ("loss", logs.loss),
            ("val_loss", logs.val_loss),
            ("accuracy", logs.accuracy),
            ("val_accuracy", logs.val_accuracy),
            ("precision", logs.precision),
            ("val_precision", logs.val_precision),
        ];

        let mut parts: Vec<String> = Vec::new();
        for (name, value) in ordered_metrics {
            if let Some(v) = value {
                parts.push(format!("{}={:.4}", name, v));
            }
        }

        println!(
            "{} {}/{} - {}",
            bold("Epoch"),
            paint(&format!("{:02}", epoch_display), Tone::Accent),
            self.total_epochs,
            parts.join(" - ")
        );
    }
}

/// Keras-style callback trait for train/eval/predict lifecycle hooks.
///
/// Implement [`Callback::should_stop`] to support early stopping.
pub trait Callback {
    fn on_train_begin(&mut self) {}
    fn on_train_end(&mut self) {}

    fn on_epoch_begin(&mut self, _epoch: usize) {}
    fn on_epoch_end(&mut self, _epoch: usize, _logs: Option<&CallbackLogs>) {}

    /// Called at the end of each epoch with mutable access to the network so
    /// callbacks (e.g. early stopping) can snapshot or restore model weights.
    fn on_epoch_end_network(
        &mut self,
        _epoch: usize,
        _logs: Option<&CallbackLogs>,
        _network: &mut Network,
    ) {
    }

    fn on_batch_begin(&mut self, _batch: usize) {}
    fn on_batch_end(&mut self, _batch: usize, _logs: Option<&CallbackLogs>) {}

    fn on_predict_begin(&mut self) {}
    fn on_predict_end(&mut self) {}

    fn on_test_begin(&mut self) {}
    fn on_test_end(&mut self) {}

    // Allows callbacks like EarlyStopping to request stopping the training loop.
    fn should_stop(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{Callback, CallbackLogs, ProgressLogger};

    #[test]
    fn progress_logger_on_epoch_end_with_logs_does_not_panic() {
        let mut logger = ProgressLogger::new(10);
        let logs = CallbackLogs {
            loss: Some(0.5),
            val_loss: Some(0.4),
            accuracy: Some(0.8),
            val_accuracy: Some(0.85),
            precision: Some(0.75),
            val_precision: Some(0.78),
        };
        // Should not panic, just print
        logger.on_epoch_end(4, Some(&logs));
    }

    #[test]
    fn progress_logger_on_epoch_end_without_logs_does_not_panic() {
        let mut logger = ProgressLogger::new(5);
        logger.on_epoch_end(0, None);
    }

    #[test]
    fn progress_logger_on_epoch_end_partial_logs() {
        let mut logger = ProgressLogger::new(3);
        let logs = CallbackLogs {
            loss: Some(0.3),
            ..CallbackLogs::default()
        };
        logger.on_epoch_end(0, Some(&logs));
    }

    struct NoopCallback;
    impl Callback for NoopCallback {}

    #[test]
    fn callback_default_implementations_do_not_panic() {
        let mut cb = NoopCallback;
        cb.on_train_begin();
        cb.on_train_end();
        cb.on_epoch_begin(0);
        cb.on_epoch_end(0, None);
        cb.on_batch_begin(0);
        cb.on_batch_end(0, None);
        cb.on_predict_begin();
        cb.on_predict_end();
        cb.on_test_begin();
        cb.on_test_end();
        assert!(!cb.should_stop());
    }
}
