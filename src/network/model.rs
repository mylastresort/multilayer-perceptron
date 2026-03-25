use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    network::{callbacks::Callback, layer::Layer},
    training::{loss::LossFunction, metrics::Metrics, optimizer::OptimizerType, trainer::Trainer},
};

// Defines the Network struct, which represents the entire neural network, including its layers and learning rate.
pub struct Network {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
}

pub struct NetworkBuilder {
    layers: Vec<Layer>,
    learning_rate: f64,
}

pub trait AsInput2D<'a> {
    fn as_input_view(self) -> ArrayView2<'a, f64>;
}

impl<'a> AsInput2D<'a> for ArrayView2<'a, f64> {
    fn as_input_view(self) -> ArrayView2<'a, f64> {
        self
    }
}

impl<'a> AsInput2D<'a> for &'a Array2<f64> {
    fn as_input_view(self) -> ArrayView2<'a, f64> {
        self.view()
    }
}

pub trait AsInput1D<'a> {
    fn as_input_view(self) -> ArrayView1<'a, f64>;
}

impl<'a> AsInput1D<'a> for ArrayView1<'a, f64> {
    fn as_input_view(self) -> ArrayView1<'a, f64> {
        self
    }
}

impl<'a> AsInput1D<'a> for &'a Array1<f64> {
    fn as_input_view(self) -> ArrayView1<'a, f64> {
        self.view()
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.01,
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

    pub fn build(self) -> Network {
        assert!(
            self.layers.len() >= 3,
            "Network requires at least 2 hidden layers plus 1 output layer; got {} layer(s)",
            self.layers.len()
        );

        Network {
            layers: self.layers,
            learning_rate: self.learning_rate,
        }
    }
}

impl Network {
    // Constructor for creating a new network with a specified learning rate and an empty list of layers.
    pub fn new() -> NetworkBuilder {
        NetworkBuilder::default()
    }

    // Performs the forward pass through the
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
        batch_size: usize,
        epochs: usize,
        optimizer: OptimizerType,
        loss_fn: LossFunction,
    ) -> Metrics
    where
        IX: AsInput2D<'data>,
        IY: AsInput1D<'data>,
    {
        let mut no_callbacks: Vec<&mut dyn Callback> = Vec::new();
        self.fit_with_callbacks(
            input,
            target,
            validation_data,
            batch_size,
            epochs,
            optimizer,
            loss_fn,
            &mut no_callbacks,
        )
    }

    pub fn fit_with_callbacks<'data, IX, IY>(
        &mut self,
        input: IX,
        target: IY,
        validation_data: Option<(ArrayView2<'data, f64>, ArrayView1<'data, f64>)>,
        batch_size: usize,
        epochs: usize,
        optimizer: OptimizerType,
        loss_fn: LossFunction,
        callbacks: &mut [&mut dyn Callback],
    ) -> Metrics
    where
        IX: AsInput2D<'data>,
        IY: AsInput1D<'data>,
    {
        let mut trainer = Trainer::new(
            optimizer.create(self.learning_rate),
            loss_fn,
            batch_size,
            epochs,
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
        let _ = Network::new()
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
        let network = Network::new()
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
}
