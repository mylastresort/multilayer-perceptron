use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::data::preprocessing::StandardScaler;
use crate::network::callbacks::Callback;
use crate::network::layer::Layer;
use crate::training::loss::LossFunction;
use crate::training::metrics::Metrics;
use crate::training::optimizer::OptimizerType;
use crate::training::trainer::Trainer;

pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub scaler: Option<StandardScaler>,
    pub loss: LossFunction,
}

pub struct NetworkBuilder {
    layers: Vec<Layer>,
    learning_rate: f64,
    scaler: Option<StandardScaler>,
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
            scaler: None,
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
            scaler: self.scaler,
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

    /// Number of outputs: width of the last layer (1 or 2 for BCE).
    pub fn output_size(&self) -> usize {
        self.layers.last().map_or(0, |l| l.weights.ncols())
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
