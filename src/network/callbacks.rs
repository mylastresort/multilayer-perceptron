#[derive(Debug, Clone, Default)]
pub struct CallbackLogs {
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub val_accuracy: Option<f64>,
}

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
        let loss = logs.loss.unwrap_or(f64::NAN);
        let val_loss = logs.val_loss.unwrap_or(f64::NAN);
        let accuracy = logs.accuracy.unwrap_or(f64::NAN);
        let val_accuracy = logs.val_accuracy.unwrap_or(f64::NAN);

        println!(
            "Display progress: epoch {:02}/{} - loss: {:.4} - val_loss: {:.4} - accuracy: {:.4} - val_accuracy: {:.4}",
            epoch_display, self.total_epochs, loss, val_loss, accuracy, val_accuracy
        );
    }
}

// Keras-style callback placeholder API for train/eval/predict lifecycle hooks.
pub trait Callback {
    fn on_train_begin(&mut self) {}
    fn on_train_end(&mut self) {}

    fn on_epoch_begin(&mut self, _epoch: usize) {}
    fn on_epoch_end(&mut self, _epoch: usize, _logs: Option<&CallbackLogs>) {}

    fn on_batch_begin(&mut self, _batch: usize) {}
    fn on_batch_end(&mut self, _batch: usize, _logs: Option<&CallbackLogs>) {}

    fn on_predict_begin(&mut self) {}
    fn on_predict_end(&mut self) {}

    fn on_test_begin(&mut self) {}
    fn on_test_end(&mut self) {}
}
