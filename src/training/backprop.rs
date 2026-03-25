use ndarray::{Array1, Array2};

use crate::network::model::Network;

#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl From<(Array2<f64>, Array1<f64>)> for LayerGradients {
    fn from((weights, bias): (Array2<f64>, Array1<f64>)) -> Self {
        Self { weights, bias }
    }
}

impl Network {
    pub fn backward(&mut self, loss_gradient: Array2<f64>) -> Vec<LayerGradients> {
        let (_, mut gradients) = self.layers.iter().rev().fold(
            (loss_gradient, Vec::new()),
            |(_gr, mut _gradients), layer| {
                let (grad_input, grad_weights, grad_bias) = layer.backward(&_gr);
                _gradients.push(LayerGradients::from((grad_weights, grad_bias)));
                (grad_input, _gradients)
            },
        );

        gradients.reverse();
        gradients
    }
}
