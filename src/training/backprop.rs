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

#[cfg(test)]
mod tests {
    use super::LayerGradients;
    use crate::network::{
        activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
        model::Network,
    };
    use ndarray::{Array1, Array2, arr2};

    fn tiny_net() -> Network {
        Network::new()
            .add_layer(Layer::new(2, 4, ActivationFunction::Sigmoid, WeightInitializer::He))
            .add_layer(Layer::new(4, 4, ActivationFunction::Sigmoid, WeightInitializer::He))
            .add_layer(Layer::new(4, 1, ActivationFunction::Sigmoid, WeightInitializer::He))
            .build()
    }

    #[test]
    fn layer_gradients_from_tuple() {
        let w = Array2::zeros((3, 4));
        let b = Array1::zeros(4);
        let lg = LayerGradients::from((w.clone(), b.clone()));
        assert_eq!(lg.weights.dim(), w.dim());
        assert_eq!(lg.bias.len(), b.len());
    }

    #[test]
    fn backward_returns_one_gradient_per_layer() {
        let mut net = tiny_net();
        let input = arr2(&[[0.5, 0.3]]);
        let _ = net.forward(input.view()); // populate caches
        let loss_grad = Array2::ones((1, 1));
        let grads = net.backward(loss_grad);
        assert_eq!(grads.len(), net.layers.len());
    }

    #[test]
    fn backward_gradient_shapes_match_layer_weights() {
        let mut net = tiny_net();
        let input = arr2(&[[0.5, 0.3]]);
        let _ = net.forward(input.view());
        let loss_grad = Array2::ones((1, 1));
        let grads = net.backward(loss_grad);
        for (layer, grad) in net.layers.iter().zip(grads.iter()) {
            assert_eq!(grad.weights.dim(), layer.weights.dim());
            assert_eq!(grad.bias.len(), layer.bias.len());
        }
    }
}
