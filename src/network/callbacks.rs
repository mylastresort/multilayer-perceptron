use crate::console::{Tone, bold, paint};

#[derive(Debug, Clone, Default)]
pub struct CallbackLogs {
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub val_precision: Option<f64>,
    pub recall: Option<f64>,
    pub val_recall: Option<f64>,
    pub f1: Option<f64>,
    pub val_f1: Option<f64>,
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

        let ordered_metrics: [(&str, Option<f64>); 10] = [
            ("loss", logs.loss),
            ("val_loss", logs.val_loss),
            ("accuracy", logs.accuracy),
            ("val_accuracy", logs.val_accuracy),
            ("precision", logs.precision),
            ("val_precision", logs.val_precision),
            ("recall", logs.recall),
            ("val_recall", logs.val_recall),
            ("f1", logs.f1),
            ("val_f1", logs.val_f1),
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

    // Allows callbacks like EarlyStopping to request stopping the training loop.
    fn should_stop(&self) -> bool {
        false
    }
}
