use mlp::network::callbacks::{Callback, CallbackLogs, ProgressLogger};

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
    assert!(!cb.should_stop());
}
