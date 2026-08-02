use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    network::{callbacks::Callback, layer::Layer},
    training::{loss::LossFunction, metrics::Metrics, optimizer::OptimizerType, trainer::Trainer},
};

pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub feature_mean: Option<ndarray::Array1<f64>>,
    pub feature_std: Option<ndarray::Array1<f64>>,
    pub loss: LossFunction,
}

pub struct NetworkBuilder {
    layers: Vec<Layer>,
    learning_rate: f64,
    feature_mean: Option<ndarray::Array1<f64>>,
    feature_std: Option<ndarray::Array1<f64>>,
    loss: LossFunction,
}

pub trait AsInput2D<'a> {
    fn as_input_view(&self) -> ArrayView2<'a, f64>;
}

impl<'a> AsInput2D<'a> for ArrayView2<'a, f64> {
    fn as_input_view(&self) -> ArrayView2<'a, f64> {
        *self
    }
}

impl<'a> AsInput2D<'a> for &'a Array2<f64> {
    fn as_input_view(&self) -> ArrayView2<'a, f64> {
        self.view()
    }
}

pub trait AsInput1D<'a> {
    fn as_input_view(&self) -> ArrayView1<'a, f64>;
}

impl<'a> AsInput1D<'a> for ArrayView1<'a, f64> {
    fn as_input_view(&self) -> ArrayView1<'a, f64> {
        *self
    }
}

impl<'a> AsInput1D<'a> for &'a Array1<f64> {
    fn as_input_view(&self) -> ArrayView1<'a, f64> {
        self.view()
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.01,
            feature_mean: None,
            feature_std: None,
            loss: LossFunction::default(),
        }
    }
}

impl NetworkBuilder {
    pub fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    pub fn build(self) -> Network {
        assert!(
            self.layers.len() >= 3,
            "Network requires at least 2 hidden layers plus 1 output layer; got {} layer(s)",
            self.layers.len()
        );

        Network {
            layers: self.layers,
            learning_rate: self.learning_rate,
            feature_mean: self.feature_mean,
            feature_std: self.feature_std,
            loss: self.loss,
        }
    }
}

pub struct FitConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub optimizer: OptimizerType,
    pub loss_fn: LossFunction,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::default()
    }

    pub fn forward<'a, I>(&mut self, input: I) -> Array2<f64>
    where
        I: AsInput2D<'a>,
    {
        let mut output = input.as_input_view().to_owned();
        for layer in &mut self.layers {
            output = layer.forward(output.view());
        }
        output
    }

    pub fn fit<'data, IX, IY>(
        &mut self,
        input: IX,
        target: IY,
        validation_data: Option<(ArrayView2<'data, f64>, ArrayView1<'data, f64>)>,
        config: FitConfig,
    ) -> Metrics
    where
        IX: AsInput2D<'data>,
        IY: AsInput1D<'data>,
    {
        let mut no_callbacks: Vec<&mut dyn Callback> = Vec::new();
        self.fit_with_callbacks(input, target, validation_data, config, &mut no_callbacks)
    }

    pub fn fit_with_callbacks<'data, IX, IY>(
        &mut self,
        input: IX,
        target: IY,
        validation_data: Option<(ArrayView2<'data, f64>, ArrayView1<'data, f64>)>,
        config: FitConfig,
        callbacks: &mut [&mut dyn Callback],
    ) -> Metrics
    where
        IX: AsInput2D<'data>,
        IY: AsInput1D<'data>,
    {
        let mut trainer = Trainer::new(
            config.optimizer.create(self.learning_rate),
            config.loss_fn,
            config.batch_size,
            config.epochs,
        );

        trainer.train(
            self,
            input.as_input_view(),
            target.as_input_view(),
            validation_data,
            callbacks,
        )
    }

    pub fn predict<'a, I>(&mut self, input: I) -> Array2<f64>
    where
        I: AsInput2D<'a>,
    {
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::Network;
    use crate::network::{
        activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
    };

    #[test]
    #[should_panic(expected = "at least 2 hidden layers")]
    fn builder_rejects_less_than_two_hidden_layers() {
        let _ = Network::builder()
            .add_layer(Layer::new(
                30,
                24,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                24,
                2,
                ActivationFunction::Softmax,
                WeightInitializer::He,
            ))
            .build();
    }

    #[test]
    fn builder_accepts_two_hidden_layers_and_output() {
        let network = Network::builder()
            .add_layer(Layer::new(
                30,
                24,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                24,
                24,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                24,
                2,
                ActivationFunction::Softmax,
                WeightInitializer::He,
            ))
            .build();

        assert_eq!(network.layers.len(), 3);
    }

    #[test]
    fn network_forward_output_shape_is_correct() {
        let mut net = Network::builder()
            .add_layer(Layer::new(
                4,
                8,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                8,
                8,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                8,
                2,
                ActivationFunction::Softmax,
                WeightInitializer::He,
            ))
            .build();

        let input = ndarray::Array2::zeros((5, 4));
        let output = net.forward(input.view());
        assert_eq!(output.dim(), (5, 2));
    }

    #[test]
    fn network_predict_matches_forward() {
        let mut net = Network::builder()
            .add_layer(Layer::new(
                2,
                4,
                ActivationFunction::ReLU,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::ReLU,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .build();

        let input = ndarray::Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64 * 0.1);
        let out_fwd = net.forward(input.view());
        let out_pred = net.predict(input.view());
        assert_eq!(out_fwd, out_pred);
    }

    #[test]
    fn network_learning_rate_builder_sets_field() {
        let net = Network::builder()
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
            .learning_rate(0.05)
            .build();
        assert!((net.learning_rate - 0.05).abs() < 1e-12);
    }

    #[test]
    fn network_fit_one_epoch_returns_metrics() {
        use crate::training::{loss::LossFunction, optimizer::OptimizerType};
        use ndarray::{Array1, Array2};

        let mut net = Network::builder()
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
            .build();

        let x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64 * 0.1);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let metrics = net.fit(
            x.view(),
            y.view(),
            None,
            super::FitConfig {
                batch_size: 4,
                epochs: 1,
                optimizer: OptimizerType::SGD,
                loss_fn: LossFunction::BinaryCrossEntropy,
            },
        );
        assert!(metrics.train_loss.is_finite());
    }

    #[test]
    fn network_fit_accepts_array1_reference_as_target() {
        use crate::training::{loss::LossFunction, optimizer::OptimizerType};
        use ndarray::{Array1, Array2};

        let mut net = Network::builder()
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
            .build();

        let x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64 * 0.1);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let metrics = net.fit(
            x.view(),
            &y,
            None,
            super::FitConfig {
                batch_size: 4,
                epochs: 1,
                optimizer: OptimizerType::SGD,
                loss_fn: LossFunction::BinaryCrossEntropy,
            },
        );
        assert!(metrics.train_loss.is_finite());
    }
}
