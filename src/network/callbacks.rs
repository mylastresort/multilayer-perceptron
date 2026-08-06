use crate::console::{Tone, bold, paint};
use crate::network::model::Network;

#[derive(Debug, Clone, Default)]
pub struct CallbackLogs {
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub accuracy: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub val_precision: Option<f64>,
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

pub trait Callback {
    fn on_train_begin(&mut self) {}
    fn on_train_end(&mut self) {}

    fn on_epoch_begin(&mut self, _epoch: usize) {}
    fn on_epoch_end(&mut self, _epoch: usize, _logs: Option<&CallbackLogs>) {}

    fn on_epoch_end_network(
        &mut self,
        _epoch: usize,
        _logs: Option<&CallbackLogs>,
        _network: &mut Network,
    ) {
    }

    fn should_stop(&self) -> bool {
        false
    }
}
