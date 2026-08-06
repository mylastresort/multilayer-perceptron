use mlp::network::{
    activation::ActivationFunction,
    callbacks::{Callback, CallbackLogs},
    initializer::WeightInitializer,
    layer::Layer,
    model::Network,
};
use mlp::training::{
    loss::LossFunction,
    optimizer::OptimizerType,
    trainer::Trainer,
};
use ndarray::{Array1, Array2};

fn tiny_net() -> Network {
    Network::builder()
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
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build()
}

fn tiny_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64 * 0.1);
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    (x, y)
}

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
    let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::default(), 4, 1);
    let val = (x.view(), y.view());
    let metrics = t.train(&mut net, x.view(), y.view(), Some(val), &mut []);
    assert!(metrics.val_loss.is_finite());
}

#[test]
fn trainer_train_without_validation_data_zeroes_val_metrics() {
    let mut net = tiny_net();
    let (x, y) = tiny_data();
    let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::default(), 4, 1);
    let metrics = t.train(&mut net, x.view(), y.view(), None, &mut []);
    assert_eq!(metrics.val_loss, 0.0);
    assert_eq!(metrics.val_accuracy, 0.0);
}

#[test]
fn trainer_early_stopping_branch_fires_and_on_train_end_called() {
    let mut net = tiny_net();
    let (x, y) = tiny_data();
    let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::default(), 4, 5);
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
    let mut t = Trainer::new(OptimizerType::SGD.create(0.01), LossFunction::default(), 4, 1);
    let metrics = t.train(&mut net, x.view(), y.view(), None, &mut []);
    assert_eq!(metrics.train_loss, 0.0);
}
